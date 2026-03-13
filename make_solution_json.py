import json
from pathlib import Path

# Read source files from your repo
binding = Path("solution/cuda/binding.py").read_text()
kernel = Path("solution/cuda/kernel.cu").read_text()

solution = {
    "name": "kfusion-moe-fp8-v1",
    "definition": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    "author": "KFusion",
    "spec": {
        "language": "cuda",
        "entry_point": "binding.py::kernel",
        "target_hardware": ["cuda"],
        "destination_passing_style": False,
        "binding": "tvm-ffi"
    },
    "sources": [
        {"path": "binding.py", "content": binding},
        {"path": "kernel.cu", "content": kernel},
    ],
}

Path("solution.json").write_text(json.dumps(solution, indent=2))
print("Wrote solution.json")