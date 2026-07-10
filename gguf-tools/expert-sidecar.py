#!/usr/bin/env python3
"""Build and verify an aligned routed-expert sidecar for a GGUF model.

The sidecar preserves each expert's quantized bytes and stores its gate, up, and
down projections in one aligned record. Construction is resumable and the final
path only appears after the complete temporary file has been synced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"DS4EXPT1"
VERSION = 1
HEADER_BYTES = 4096
ENTRY_BYTES = 128
HEADER_FORMAT = struct.Struct("<8sIIIIIIQQQQQQIII32s32s")
ENTRY_FORMAT = struct.Struct("<II8Q32sII16s")
FLAG_COMPLETE = 1
ENTRY_COMPLETE = 1
COPY_BYTES = 4 << 20

# GGML type -> (elements per block, bytes per block).
TYPE_INFO = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
    7: (32, 24), 8: (32, 34), 9: (32, 40), 10: (256, 84),
    11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98),
    19: (256, 110), 20: (256, 50), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8),
    28: (1, 8), 29: (256, 56), 30: (1, 2),
}
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}
PATTERNS = [
    re.compile(r"(?:blk|layers)\.(\d+)\.ffn_(gate|up|down)_exps\.weight$"),
    re.compile(r"layers\.(\d+)\.ffn\.experts\.(w1|w2|w3)\.weight$"),
]
PART = {"gate": "gate", "up": "up", "down": "down",
        "w1": "gate", "w3": "up", "w2": "down"}


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class Reader:
    def __init__(self, file):
        self.file = file

    def read(self, size: int) -> bytes:
        data = self.file.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8")

    def value(self, kind: int, keep: bool = False, depth: int = 0):
        if depth > 8:
            raise ValueError("metadata nesting too deep")
        if kind in SCALAR_BYTES:
            data = self.read(SCALAR_BYTES[kind])
            return int.from_bytes(data, "little") if keep else None
        if kind == 8:
            value = self.string()
            return value if keep else None
        if kind == 9:
            item_kind = self.u32()
            count = self.u64()
            values = [self.value(item_kind, keep, depth + 1) for _ in range(count)]
            return values if keep else None
        raise ValueError(f"unknown metadata type {kind}")


@dataclass
class Tensor:
    name: str
    dims: list[int]
    kind: int
    relative_offset: int
    size: int
    absolute_offset: int = 0


@dataclass
class Record:
    layer: int
    expert: int
    offset: int
    sources: tuple[int, int, int]
    sizes: tuple[int, int, int]
    digest: bytes = b""

    @property
    def payload_bytes(self) -> int:
        return sum(self.sizes)

    def entry(self, complete: bool = True) -> bytes:
        gate_offset = self.offset
        up_offset = gate_offset + self.sizes[0]
        down_offset = up_offset + self.sizes[1]
        digest = self.digest if complete else bytes(32)
        return ENTRY_FORMAT.pack(
            self.layer, self.expert, self.offset, self.payload_bytes,
            gate_offset, self.sizes[0], up_offset, self.sizes[1],
            down_offset, self.sizes[2], digest,
            ENTRY_COMPLETE if complete else 0, 0, bytes(16),
        )


def parse_gguf(path: Path):
    model_size = path.stat().st_size
    with path.open("rb") as file:
        reader = Reader(file)
        if reader.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = reader.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = reader.u64()
        metadata_count = reader.u64()
        alignment = 32
        for _ in range(metadata_count):
            key = reader.string()
            kind = reader.u32()
            value = reader.value(kind, key == "general.alignment")
            if key == "general.alignment":
                alignment = int(value)
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"invalid GGUF alignment {alignment}")

        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = [reader.u64() for _ in range(reader.u32())]
            kind = reader.u32()
            relative_offset = reader.u64()
            if kind not in TYPE_INFO:
                raise ValueError(f"unsupported tensor type {kind}: {name}")
            elements = 1
            for dimension in dimensions:
                elements *= dimension
            block_elements, block_bytes = TYPE_INFO[kind]
            if not dimensions or dimensions[0] % block_elements:
                raise ValueError(f"tensor row is not quantization-block aligned: {name}")
            size = elements // block_elements * block_bytes
            tensors.append(Tensor(name, dimensions, kind, relative_offset, size))
        data_offset = align(file.tell(), alignment)

    for tensor in tensors:
        tensor.absolute_offset = data_offset + tensor.relative_offset
        if tensor.absolute_offset + tensor.size > model_size:
            raise ValueError(f"tensor extends beyond end of GGUF: {tensor.name}")
    return version, alignment, data_offset, tensors


def routed_tensors(tensors: list[Tensor]):
    groups = {}
    for tensor in tensors:
        match = None
        for pattern in PATTERNS:
            match = pattern.search(tensor.name)
            if match:
                break
        if not match:
            continue
        layer = int(match.group(1))
        part = PART[match.group(2)]
        if len(tensor.dims) != 3:
            raise ValueError(f"routed tensor is not 3D: {tensor.name}")
        experts = tensor.dims[2]
        if not experts or tensor.size % experts:
            raise ValueError(f"tensor is not divisible by experts: {tensor.name}")
        layer_parts = groups.setdefault(layer, {})
        if part in layer_parts:
            raise ValueError(f"duplicate routed {part} tensor in layer {layer}")
        layer_parts[part] = (tensor, tensor.size // experts, experts)

    if not groups:
        raise ValueError("no routed expert tensors found")
    for layer, parts in groups.items():
        if set(parts) != {"gate", "up", "down"}:
            raise ValueError(f"layer {layer} lacks gate/up/down")
        if len({value[2] for value in parts.values()}) != 1:
            raise ValueError(f"layer {layer} expert counts differ")
    return groups


def make_layout(groups, alignment: int):
    count = sum(next(iter(parts.values()))[2] for parts in groups.values())
    position = align(HEADER_BYTES + count * ENTRY_BYTES, alignment)
    records = []
    for layer in sorted(groups):
        parts = groups[layer]
        experts = next(iter(parts.values()))[2]
        for expert in range(experts):
            position = align(position, alignment)
            sources, sizes = [], []
            for part in ("gate", "up", "down"):
                tensor, size, _ = parts[part]
                sources.append(tensor.absolute_offset + expert * size)
                sizes.append(size)
            records.append(Record(layer, expert, position,
                                  tuple(sources), tuple(sizes)))
            position += sum(sizes)
    return records, align(position, alignment)


def sha256_file(path: Path) -> bytes:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while data := file.read(16 << 20):
            digest.update(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("model changed while hashing")
    return digest.digest()


def parse_hash(value: str | None, model: Path) -> bytes:
    if value is None:
        return sha256_file(model)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("model SHA-256 must be 64 hex digits")
    return bytes.fromhex(value)


def header_bytes(model: Path, model_digest: bytes, records: list[Record],
                 alignment: int, total_size: int, complete: bool,
                 directory_digest: bytes = bytes(32)) -> bytes:
    stat = model.stat()
    first_layer = min(record.layer for record in records)
    last_layer = max(record.layer for record in records)
    layer_count = len({record.layer for record in records})
    payload_offset = align(HEADER_BYTES + len(records) * ENTRY_BYTES, alignment)
    packed = HEADER_FORMAT.pack(
        MAGIC, VERSION, HEADER_BYTES, ENTRY_BYTES, alignment,
        FLAG_COMPLETE if complete else 0, 0, stat.st_size, stat.st_mtime_ns,
        len(records), HEADER_BYTES, payload_offset, total_size,
        layer_count, first_layer, last_layer, model_digest, directory_digest,
    )
    return packed + bytes(HEADER_BYTES - len(packed))


def unpack_header(data: bytes) -> dict:
    if len(data) < HEADER_BYTES:
        raise ValueError("truncated sidecar header")
    values = HEADER_FORMAT.unpack_from(data)
    keys = ("magic", "version", "header_bytes", "entry_bytes", "alignment",
            "flags", "reserved", "model_bytes", "model_mtime_ns", "records",
            "directory_offset", "payload_offset", "sidecar_bytes", "layers",
            "first_layer", "last_layer", "model_sha256", "directory_sha256")
    header = dict(zip(keys, values))
    if header["magic"] != MAGIC or header["version"] != VERSION:
        raise ValueError("not a supported DS4 expert sidecar")
    if header["header_bytes"] != HEADER_BYTES or header["entry_bytes"] != ENTRY_BYTES:
        raise ValueError("unsupported sidecar header or entry size")
    return header


def pread_exact(fd: int, size: int, offset: int) -> bytes:
    chunks = []
    read_bytes = 0
    while read_bytes < size:
        data = os.pread(fd, size - read_bytes, offset + read_bytes)
        if not data:
            raise ValueError("short read")
        chunks.append(data)
        read_bytes += len(data)
    return b"".join(chunks)


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    written = 0
    while written < len(data):
        count = os.pwrite(fd, data[written:], offset + written)
        if count <= 0:
            raise OSError("short write")
        written += count


def copy_record(model_fd: int, sidecar_fd: int, record: Record) -> bytes:
    digest = hashlib.sha256()
    destination = record.offset
    for source, size in zip(record.sources, record.sizes):
        copied = 0
        while copied < size:
            amount = min(COPY_BYTES, size - copied)
            data = pread_exact(model_fd, amount, source + copied)
            digest.update(data)
            pwrite_all(sidecar_fd, data, destination + copied)
            copied += amount
        destination += size
    return digest.digest()


def completed_entries(fd: int, records: list[Record]) -> int:
    completed = 0
    for index, record in enumerate(records):
        data = pread_exact(fd, ENTRY_BYTES, HEADER_BYTES + index * ENTRY_BYTES)
        if len(data) != ENTRY_BYTES or data == bytes(ENTRY_BYTES):
            break
        values = ENTRY_FORMAT.unpack(data)
        if values[11] != ENTRY_COMPLETE:
            break
        digest = values[10]
        record.digest = digest
        if data != record.entry():
            raise ValueError(f"temporary sidecar directory mismatch at record {index}")
        completed += 1
    return completed


def checkpoint(fd: int, pending: list[tuple[int, Record]]) -> None:
    if not pending:
        return
    # Make payload durable before publishing completed directory entries.
    os.fsync(fd)
    for index, record in pending:
        pwrite_all(fd, record.entry(), HEADER_BYTES + index * ENTRY_BYTES)
    os.fsync(fd)
    pending.clear()


def build(model: Path, output: Path, records: list[Record], alignment: int,
          total_size: int, model_digest: bytes, resume: bool,
          checkpoint_records: int) -> None:
    temporary = output.with_name(output.name + ".part")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if temporary.exists() and not resume:
        raise FileExistsError(f"temporary output exists (use --resume): {temporary}")

    flags = os.O_RDWR | os.O_CREAT
    if not resume:
        flags |= os.O_EXCL
    model_stat = model.stat()
    model_identity = (model_stat.st_size, model_stat.st_mtime_ns)
    sidecar_fd = os.open(temporary, flags, 0o644)
    model_fd = os.open(model, os.O_RDONLY)
    try:
        if os.fstat(sidecar_fd).st_size == 0:
            free_bytes = os.statvfs(temporary.parent).f_bavail * os.statvfs(temporary.parent).f_frsize
            if free_bytes < total_size:
                raise OSError(f"sidecar needs {total_size} bytes but only {free_bytes} are free")
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(sidecar_fd, 0, total_size)
            else:
                os.ftruncate(sidecar_fd, total_size)
            pwrite_all(sidecar_fd, header_bytes(model, model_digest, records,
                                                alignment, total_size, False), 0)
            os.fsync(sidecar_fd)
            start = 0
        else:
            raw_header = pread_exact(sidecar_fd, HEADER_BYTES, 0)
            header = unpack_header(raw_header)
            expected = header_bytes(model, model_digest, records, alignment,
                                    total_size, False)
            if header["flags"] & FLAG_COMPLETE:
                expected = header_bytes(model, model_digest, records, alignment,
                                        total_size, True, header["directory_sha256"])
            if raw_header != expected:
                raise ValueError("temporary sidecar does not match model or requested layout")
            if os.fstat(sidecar_fd).st_size != total_size:
                raise ValueError("temporary sidecar has the wrong size")
            start = completed_entries(sidecar_fd, records)
            if header["flags"] & FLAG_COMPLETE:
                if start != len(records):
                    raise ValueError("complete temporary sidecar has an incomplete directory")
                directory = pread_exact(sidecar_fd, len(records) * ENTRY_BYTES,
                                        HEADER_BYTES)
                if hashlib.sha256(directory).digest() != header["directory_sha256"]:
                    raise ValueError("complete temporary sidecar directory checksum mismatch")

        pending = []
        for index in range(start, len(records)):
            record = records[index]
            record.digest = copy_record(model_fd, sidecar_fd, record)
            pending.append((index, record))
            if len(pending) >= checkpoint_records:
                checkpoint(sidecar_fd, pending)
                print(f"copied {index + 1}/{len(records)} records", file=sys.stderr)
        checkpoint(sidecar_fd, pending)

        if (model.stat().st_size, model.stat().st_mtime_ns) != model_identity:
            raise ValueError("model changed while building sidecar")
        directory = pread_exact(sidecar_fd, len(records) * ENTRY_BYTES, HEADER_BYTES)
        directory_digest = hashlib.sha256(directory).digest()
        final_header = header_bytes(model, model_digest, records, alignment,
                                    total_size, True, directory_digest)
        pwrite_all(sidecar_fd, final_header, 0)
        os.fsync(sidecar_fd)
    finally:
        os.close(model_fd)
        os.close(sidecar_fd)

    # A hard link publishes the synced inode atomically without replacing a file
    # that raced the initial existence check.
    os.link(temporary, output)
    os.unlink(temporary)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def verify(model: Path, sidecar: Path, records: list[Record],
           model_digest: bytes) -> None:
    with sidecar.open("rb", buffering=0) as file:
        header = unpack_header(file.read(HEADER_BYTES))
        if not header["flags"] & FLAG_COMPLETE:
            raise ValueError("sidecar is not marked complete")
        if header["model_sha256"] != model_digest:
            raise ValueError("sidecar model SHA-256 mismatch")
        if header["model_bytes"] != model.stat().st_size:
            raise ValueError("sidecar model size mismatch")
        if header["records"] != len(records) or header["sidecar_bytes"] != sidecar.stat().st_size:
            raise ValueError("sidecar layout does not match model")
        directory = file.read(len(records) * ENTRY_BYTES)
        if hashlib.sha256(directory).digest() != header["directory_sha256"]:
            raise ValueError("sidecar directory checksum mismatch")

        model_fd = os.open(model, os.O_RDONLY)
        sidecar_fd = os.open(sidecar, os.O_RDONLY)
        try:
            for index, record in enumerate(records):
                values = ENTRY_FORMAT.unpack_from(directory, index * ENTRY_BYTES)
                record.digest = values[10]
                if values[11] != ENTRY_COMPLETE or record.entry() != directory[index * ENTRY_BYTES:(index + 1) * ENTRY_BYTES]:
                    raise ValueError(f"invalid sidecar directory record {index}")
                source_hash = hashlib.sha256()
                sidecar_hash = hashlib.sha256()
                destination = record.offset
                for source, size in zip(record.sources, record.sizes):
                    copied = 0
                    while copied < size:
                        amount = min(COPY_BYTES, size - copied)
                        source_hash.update(pread_exact(model_fd, amount, source + copied))
                        sidecar_hash.update(pread_exact(sidecar_fd, amount, destination + copied))
                        copied += amount
                    destination += size
                if source_hash.digest() != record.digest or sidecar_hash.digest() != record.digest:
                    raise ValueError(f"expert payload checksum mismatch at record {index}")
        finally:
            os.close(model_fd)
            os.close(sidecar_fd)


def summary(model: Path, version: int, gguf_alignment: int, data_offset: int,
            groups, records: list[Record], alignment: int, total_size: int) -> dict:
    return {
        "format": "ds4-expert-sidecar-v1", "model": str(model.resolve()),
        "model_bytes": model.stat().st_size, "gguf_version": version,
        "gguf_alignment": gguf_alignment, "tensor_data_offset": data_offset,
        "layers": len(groups), "experts": len(records), "alignment": alignment,
        "sidecar_bytes": total_size,
        "payload_bytes": sum(record.payload_bytes for record in records),
        "first_layer": min(groups), "last_layer": max(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--model-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-records", type=int, default=64)
    args = parser.parse_args()

    if args.alignment < 4096 or args.alignment & (args.alignment - 1):
        parser.error("alignment must be a power of two >= 4096")
    if args.checkpoint_records < 1:
        parser.error("--checkpoint-records must be positive")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if not args.plan and args.output is None:
        parser.error("output is required for --build and --verify")

    try:
        version, gguf_alignment, data_offset, tensors = parse_gguf(args.model)
        groups = routed_tensors(tensors)
        records, total_size = make_layout(groups, args.alignment)
        print(json.dumps(summary(args.model, version, gguf_alignment, data_offset,
                                 groups, records, args.alignment, total_size), indent=2))
        if args.plan:
            return
        model_digest = parse_hash(args.model_sha256, args.model)
        if args.build:
            build(args.model, args.output, records, args.alignment, total_size,
                  model_digest, args.resume, args.checkpoint_records)
            print(f"built {args.output}", file=sys.stderr)
        else:
            verify(args.model, args.output, records, model_digest)
            print(f"verified {args.output}", file=sys.stderr)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
