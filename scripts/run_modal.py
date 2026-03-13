"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Runs benchmarks on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

from pathlib import Path
import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

# IMPORTANT:
# The Modal volume is mounted at /data (see decorator),
# and inside it your upload created a folder named "flashinfer-trace".
# The actual dataset root (with definitions/, workloads/, etc.) is one level deeper.
TRACE_SET_PATH = "/data/flashinfer-trace"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:${PATH}",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
        }
    )
    .pip_install("flashinfer-bench", "flashinfer-python", "torch", "triton", "numpy")
)

@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={"/data": trace_volume},  # mount volume at /data
)
def run_benchmark(solution_json: str, config_dict: dict | None = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
    import os, glob

    solution = Solution.model_validate_json(solution_json)

    if config_dict is None:
        config = BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=1)  # cheap smoke test
    else:
        config = BenchmarkConfig(**config_dict)

    # Debug: verify dataset visibility
    print("LS /data:", os.listdir("/data")[:30])
    print("TRACE_SET_PATH =", TRACE_SET_PATH)
    print("LS TRACE_SET_PATH:", os.listdir(TRACE_SET_PATH)[:30])
    print("num definition json:", len(glob.glob(TRACE_SET_PATH + "/definitions/*.json")))
    print("sample definition json:", glob.glob(TRACE_SET_PATH + "/definitions/*.json")[:5])

    # Load traceset
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    print("Num definitions:", len(trace_set.definitions))
    moe_defs = [k for k in trace_set.definitions.keys() if "moe" in k]
    print("MOE defs (first 50):", moe_defs[:50])

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    workloads = workloads[:1]  # run 1 workload for cheap test

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {"status": trace.evaluation.status.value, "solution": trace.solution}
            print("EVAL RAW:", trace.evaluation)
            print("EVAL DICT:", getattr(trace.evaluation, "__dict__", None))
            # ✅ add this
            if getattr(trace.evaluation, "message", None):
                entry["message"] = trace.evaluation.message
            if getattr(trace.evaluation, "error", None):
                entry["error"] = str(trace.evaluation.error)
            if getattr(trace.evaluation, "logs", None):
                entry["logs"] = trace.evaluation.logs

            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")
            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")
            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")
            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")
            print()


@app.local_entrypoint()
def main():
    print("Reading solution.json locally...")
    solution_json = Path("solution.json").read_text()

    print("Running benchmark on Modal B200...")
    results = run_benchmark.remote(solution_json)

    print_results(results)
    print(results)