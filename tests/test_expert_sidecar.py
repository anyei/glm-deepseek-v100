#!/usr/bin/env python3
"""Small-file regression tests for gguf-tools/expert-sidecar.py."""
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "gguf-tools" / "expert-sidecar.py"


def aligned(value, alignment=32):
    return (value + alignment - 1) // alignment * alignment


def gguf_string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def make_model(path):
    names = [
        "blk.2.ffn_gate_exps.weight",
        "blk.2.ffn_up_exps.weight",
        "blk.2.ffn_down_exps.weight",
    ]
    # Q8_0: [32, 2, 2] is 136 bytes, or 68 bytes per expert.
    offsets = [0, 160, 320]
    table = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(names), 0))
    for name, offset in zip(names, offsets):
        table += gguf_string(name)
        table += struct.pack("<IQQQIQ", 3, 32, 2, 2, 8, offset)
    data_offset = aligned(len(table))
    contents = table + bytes(data_offset - len(table))
    for index, offset in enumerate(offsets):
        if len(contents) < data_offset + offset:
            contents += bytes(data_offset + offset - len(contents))
        contents += bytes([0x20 + index]) * 136
    path.write_bytes(contents)


class ExpertSidecarTest(unittest.TestCase):
    def run_tool(self, *arguments, check=True):
        return subprocess.run(
            ["python3", str(TOOL), *map(str, arguments)],
            check=check, text=True, capture_output=True,
        )

    def test_plan_build_verify_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "tiny.gguf"
            sidecar = Path(directory) / "tiny.experts"
            make_model(model)

            plan = self.run_tool(model, "--plan")
            try:
                summary = json.loads(plan.stdout)
            except json.JSONDecodeError as error:
                self.fail(f"sidecar plan did not emit valid JSON: {error}")
            self.assertEqual(summary["layers"], 1)
            self.assertEqual(summary["experts"], 2)
            self.assertEqual(summary["payload_bytes"], 408)
            self.assertEqual(summary["sidecar_bytes"], 16384)

            self.run_tool(model, sidecar, "--build", "--checkpoint-records", "1")
            self.assertTrue(sidecar.is_file())
            self.assertFalse(Path(str(sidecar) + ".part").exists())
            self.run_tool(model, sidecar, "--verify")

            # A crash after syncing the complete header but before publication
            # is recoverable without copying payloads again.
            temporary = Path(str(sidecar) + ".part")
            os.replace(sidecar, temporary)
            self.run_tool(model, sidecar, "--build", "--resume")
            self.run_tool(model, sidecar, "--verify")

            with sidecar.open("r+b") as file:
                file.seek(8192)
                byte = file.read(1)
                file.seek(8192)
                file.write(bytes([byte[0] ^ 1]))
            failed = self.run_tool(model, sidecar, "--verify", check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("payload checksum mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
