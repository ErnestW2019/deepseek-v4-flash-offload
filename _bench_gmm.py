"""Check npu_grouped_matmul accepts non-contiguous (transposed view) weights."""

import torch
import torch_npu  # noqa: F401

DEV = torch.device("npu:0")
torch.manual_seed(0)
OUT, IN, E = 8, 4, 3
counts = [2, 0, 3]
M = sum(counts)

x = torch.randn(M, IN, dtype=torch.bfloat16, device=DEV)
nks = [torch.randn(OUT, IN, dtype=torch.bfloat16, device=DEV) for _ in range(E)]  # [N,K]

ref = torch.empty(M, OUT, dtype=torch.bfloat16, device=DEV)
o = 0
for e, c in enumerate(counts):
    if c:
        ref[o:o + c] = x[o:o + c] @ nks[e].t()
        o += c
gl = torch.tensor(counts, dtype=torch.int64, device=DEV)


def run(tag, wl):
    print(f"  {tag}: contig={[w.is_contiguous() for w in wl]}")
    try:
        got = torch_npu.npu_grouped_matmul(
            x=[x], weight=wl, split_item=2, group_list_type=1,
            group_type=0, group_list=gl)[0]
        d = (got.float() - ref.float()).abs().max().item()
        print(f"    ok max_abs_diff={d:.3e}")
    except Exception as ex:
        print("    ERR", str(ex).splitlines()[0])


run("non-contiguous .t()", [w.t() for w in nks])
run("contiguous .t().contiguous()", [w.t().contiguous() for w in nks])
