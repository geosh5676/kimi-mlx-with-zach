#!/Users/Shared/.venvs/mlx/bin/python
# -*- coding: utf-8 -*-

import os, argparse
import mlx.core as mx
from mlx_lm import load, generate

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/Shared/.cache/huggingface"
os.environ["TRUST_REMOTE_CODE"] = "1"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    # Initialize distributed
    group = mx.distributed.init()
    
    if group.rank() == 0:
        print(f"[INFO] Distributed group initialized: {group.size()} nodes")
        print(f"[INFO] Loading Kimi K2 model across nodes...")

    # Load model without trust_remote_code parameter
    model, tok = load(args.model, lazy=True)
    
    # Synchronize before generation
    mx.eval(mx.zeros(1))
    
    if group.rank() == 0:
        print(f"[INFO] Generating response...")
    
    # Generate
    out = generate(
        model, tok,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=(group.rank() == 0),
    )
    
    if group.rank() == 0:
        print("\n[OUTPUT]:", out)

if __name__ == "__main__":
    main()
