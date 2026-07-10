#!/usr/bin/env python3
"""Plan the aligned routed-expert sidecar layout for a GGUF.

No tensor bytes are transformed. Each planned record stores one expert's
 gate | up | down payload contiguously and aligned for direct I/O. Construction
remains disabled until the v1 header/directory format is reviewed.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, struct, sys, time
from dataclasses import dataclass
from pathlib import Path

MAGIC=b"DS4EXPT1"; VERSION=1; HEADER=4096; ENTRY=128
TYPE_INFO={0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,40),10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),16:(256,66),17:(256,74),18:(256,98),19:(256,110),20:(256,50),21:(256,110),22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),29:(256,56),30:(1,2)}
SCALAR={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
PATTERNS=[re.compile(r"(?:blk|layers)\.(\d+)\.ffn_(gate|up|down)_exps\.weight$"),re.compile(r"layers\.(\d+)\.ffn\.experts\.(w1|w2|w3)\.weight$")]
PART={"gate":"gate","up":"up","down":"down","w1":"gate","w3":"up","w2":"down"}

def align(v,a): return (v+a-1)//a*a
class Reader:
 def __init__(self,f): self.f=f
 def read(self,n):
  b=self.f.read(n)
  if len(b)!=n: raise ValueError("truncated GGUF")
  return b
 def u32(self): return struct.unpack("<I",self.read(4))[0]
 def u64(self): return struct.unpack("<Q",self.read(8))[0]
 def string(self): return self.read(self.u64()).decode("utf-8")
 def value(self,t,keep=False,depth=0):
  if depth>8: raise ValueError("metadata nesting too deep")
  if t in SCALAR:
   b=self.read(SCALAR[t]); return int.from_bytes(b,"little") if keep else None
  if t==8: return self.string() if keep else (self.string() and None)
  if t==9:
   item=self.u32(); n=self.u64(); vals=[]
   for _ in range(n): vals.append(self.value(item,keep,depth+1))
   return vals if keep else None
  raise ValueError(f"unknown metadata type {t}")
@dataclass
class Tensor: name:str; dims:list[int]; type:int; rel:int; size:int; absolute:int=0

def parse(path:Path):
 with path.open("rb") as f:
  r=Reader(f)
  if r.read(4)!=b"GGUF": raise ValueError("not GGUF")
  ver=r.u32(); nt=r.u64(); nk=r.u64(); meta={}; alignment=32
  for _ in range(nk):
   key=r.string(); typ=r.u32(); val=r.value(typ,key in {"general.alignment"})
   if key=="general.alignment" and val: alignment=int(val)
  ts=[]
  for _ in range(nt):
   name=r.string(); nd=r.u32(); dims=[r.u64() for _ in range(nd)]; typ=r.u32(); rel=r.u64()
   if typ not in TYPE_INFO: raise ValueError(f"unsupported tensor type {typ}: {name}")
   elems=1
   for d in dims: elems*=d
   be,bb=TYPE_INFO[typ]; size=((elems+be-1)//be)*bb
   ts.append(Tensor(name,dims,typ,rel,size))
  data=align(f.tell(),alignment)
 for t in ts: t.absolute=data+t.rel
 return ver,alignment,data,ts

def routed(ts):
 out={}
 for t in ts:
  match=None
  for p in PATTERNS:
   match=p.search(t.name)
   if match: break
  if not match: continue
  layer=int(match.group(1)); part=PART[match.group(2)]
  if len(t.dims)<3: raise ValueError(f"routed tensor is not 3D: {t.name}")
  experts=t.dims[2]
  if t.size%experts: raise ValueError(f"tensor not divisible by experts: {t.name}")
  out.setdefault(layer,{})[part]=(t,t.size//experts,experts)
 if not out: raise ValueError("no routed expert tensors found")
 for layer,parts in out.items():
  if set(parts)!={"gate","up","down"}: raise ValueError(f"layer {layer} lacks gate/up/down")
  counts={v[2] for v in parts.values()}
  if len(counts)!=1: raise ValueError(f"layer {layer} expert counts differ")
 return out

def model_hash(path):
 h=hashlib.sha256()
 with path.open("rb",buffering=0) as f:
  while b:=f.read(16<<20): h.update(b)
 return h.hexdigest()
def layout(groups,alignment):
 records=[]; count=sum(next(iter(p.values()))[2] for p in groups.values())
 pos=align(HEADER+count*ENTRY,alignment)
 for layer in sorted(groups):
  parts=groups[layer]; n=next(iter(parts.values()))[2]
  for expert in range(n):
   pos=align(pos,alignment); rec={"layer":layer,"expert":expert,"offset":pos}
   for part in ("gate","up","down"):
    t,size,_=parts[part]; rec[part+"_bytes"]=size; rec[part+"_source"]=t.absolute+expert*size; pos+=size
   rec["payload_bytes"]=sum(rec[p+"_bytes"] for p in ("gate","up","down")); records.append(rec)
 return records,align(pos,alignment)
def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("model",type=Path); ap.add_argument("output",type=Path,nargs="?")
 ap.add_argument("--alignment",type=int,default=4096); ap.add_argument("--model-sha256"); ap.add_argument("--plan",action="store_true")
 args=ap.parse_args()
 if args.alignment<4096 or args.alignment&(args.alignment-1): ap.error("alignment must be a power of two >=4096")
 ver,ga,data,ts=parse(args.model); groups=routed(ts); records,total=layout(groups,args.alignment)
 summary={"format":"ds4-expert-sidecar-v1","model":str(args.model.resolve()),"model_bytes":args.model.stat().st_size,"gguf_version":ver,"gguf_alignment":ga,"tensor_data_offset":data,"layers":len(groups),"experts":len(records),"alignment":args.alignment,"sidecar_bytes":total,"payload_bytes":sum(r["payload_bytes"] for r in records),"first_layer":min(groups),"last_layer":max(groups)}
 print(json.dumps(summary,indent=2))
 if args.plan: return
 if not args.output: ap.error("output is required unless --plan is used")
 sha=args.model_sha256 or model_hash(args.model)
 if not re.fullmatch(r"[0-9a-fA-F]{64}",sha): ap.error("model SHA-256 must be 64 hex digits")
 # Construction is intentionally added in the next gated step after format review.
 raise SystemExit("format plan validated; construction path not enabled yet")
if __name__=="__main__": main()
