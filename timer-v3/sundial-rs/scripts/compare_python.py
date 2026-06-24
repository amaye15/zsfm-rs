#!/usr/bin/env python3
"""Compare Sundial-rs forecast against Python reference.

The flow head is stochastic; we zero-init noise in both implementations
so that the Euler trajectory is deterministic.

Usage:
    python scripts/compare_python.py [--gguf PATH] [--dtype f32|f16|q8] [--horizon 96]
"""
import argparse
import json
import math
import subprocess
import sys
import numpy as np

RUST_BIN = "./target/release/sundial-rs"


def sine_context(n=128):
    return [math.sin(2 * math.pi * i / 32) for i in range(n)]


def run_rust(gguf_path, context, horizon):
    request = json.dumps({"context": list(context), "horizon": horizon})
    result = subprocess.run(
        [RUST_BIN, "infer", "--gguf", gguf_path],
        input=request, capture_output=True, text=True, check=True,
    )
    fc = json.loads(result.stdout)["choices"][0]["forecast"]
    return np.array(fc["point"][:horizon], dtype=np.float32)


def run_python(context, horizon):
    """Run the Python Sundial reference model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("transformers or torch not available; skipping Python reference.", file=sys.stderr)
        return None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m", trust_remote_code=True)
        model.eval()

        # Patch flow_loss.sample to use zeros noise and Heun's method (matches Rust).
        def heun_sample(self_fl, z, num_samples=1):
            z = z.repeat(num_samples, 1)
            noise = torch.zeros(z.shape[0], self_fl.in_channels).to(z.device)
            x = noise.clone()
            n = self_fl.num_sampling_steps
            dt = 1.0 / n
            for i in range(n):
                t1 = (torch.ones(x.shape[0]) * i / n).to(x.device)
                k1 = self_fl.net(x, t1 * 1000, z)
                x_pred = x + k1 * dt
                t2 = (torch.ones(x.shape[0]) * (i + 1) / n).to(x.device)
                k2 = self_fl.net(x_pred, t2 * 1000, z)
                x = x + (k1 + k2) * 0.5 * dt
            x = x.reshape(num_samples, -1, self_fl.in_channels).transpose(0, 1)
            return x
        import types
        model.flow_loss.sample = types.MethodType(heun_sample, model.flow_loss)

        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)  # [1, n]
        with torch.no_grad():
            out = model(input_ids=ctx_tensor, revin=True, num_samples=1,
                        max_output_length=horizon, use_cache=False)
        # out is a tuple; first element is predictions: [1, num_samples, output_len]
        return out[0][0, 0].numpy()[:horizon]
    except Exception as e:
        print(f"Python model failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default="gguf/sundial-f32.gguf")
    parser.add_argument("--dtype", default="f32")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=96)
    args = parser.parse_args()

    gguf = args.gguf if args.dtype == "f32" else f"gguf/sundial-{args.dtype}.gguf"
    context = sine_context(args.context_len)

    print(f"Context length: {len(context)}, horizon: {args.horizon}")
    print(f"Running Rust inference ({gguf}) …")
    rust_out = run_rust(gguf, context, args.horizon)
    print(f"  Rust output shape: {rust_out.shape}, first 5: {rust_out[:5]}")

    print("Running Python reference …")
    py_out = run_python(context, args.horizon)
    if py_out is None:
        print("Cannot compare without Python reference.")
        print("\nRust output (first 10):")
        print(rust_out[:10])
        return

    n = min(len(rust_out), len(py_out))
    rust_out = rust_out[:n]
    py_out = py_out[:n]
    diff = np.abs(rust_out - py_out)
    print(f"\nComparison over {n} steps:")
    print(f"  Max  |Δ|: {diff.max():.6f}")
    print(f"  Mean |Δ|: {diff.mean():.6f}")
    print(f"\nFirst 8 steps:")
    print(f"  Python: {' '.join(f'{v:.4f}' for v in py_out[:8])}")
    print(f"    Rust: {' '.join(f'{v:.4f}' for v in rust_out[:8])}")
    print(f"    Diff: {' '.join(f'{abs(a-b):.4f}' for a, b in zip(py_out[:8], rust_out[:8]))}")


if __name__ == "__main__":
    main()
