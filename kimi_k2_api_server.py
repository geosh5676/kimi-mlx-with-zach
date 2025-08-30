from flask import Flask, request, jsonify
import subprocess
import json
import re
import os
import sys
import shlex
from copy import deepcopy

app = Flask(__name__)


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json or {}
    messages = data.get('messages', [])
    max_tokens = int(data.get('max_tokens', 10000))

    # Combine all messages into a complete prompt for Kimi K2
    prompt = ""
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'system':
            prompt += f"System: {content}\n\n"
        elif role == 'user':
            if prompt and not prompt.endswith('\n\n'):
                prompt += "\n\n"
            prompt += f"User: {content}"
        elif role == 'assistant':
            prompt += f"\n\nAssistant: {content}"
    
    # Ensure prompt ends cleanly
    prompt = prompt.strip()

    # Launch distributed Kimi K2 via mlx
    # Prefer an explicit launcher path; fall back to PATH discovery
    launch_override = os.getenv('MLX_LAUNCH')
    default_mlx_bin = os.getenv('MLX_VENV_BIN', '/Users/Shared/.venvs/mlx/bin')
    default_launcher = os.path.join(default_mlx_bin, 'mlx.launch')

    if launch_override:
        launch_parts = shlex.split(launch_override)
        cmd = launch_parts
    elif os.path.isfile(default_launcher) and os.access(default_launcher, os.X_OK):
        cmd = [default_launcher]
    else:
        # Last resort: rely on PATH for 'mlx.launch'
        cmd = ['mlx.launch']

    cmd = cmd + [
        '--hostfile', '/Users/Shared/hosts.json',
        '/Users/Shared/pipeline_generate.py',
        '--model', 'mlx-community/Kimi-K2-Instruct-4bit',
        '--prompt', prompt,
        '--max-tokens', str(max_tokens)
    ]
    
    # Conditional debug logging to avoid blocking I/O with large prompts
    prompt_len = len(prompt)
    if prompt_len < 50000:  # Only detailed debug for reasonable sizes
        try:
            print(f"[DEBUG] Processing prompt - Length: {prompt_len} chars, Max tokens: {max_tokens}", flush=True)
            if prompt_len > 200 and prompt_len < 10000:
                print(f"[DEBUG] Prompt preview (first 200 chars): {prompt[:200]}...", flush=True)
            elif prompt_len <= 200:
                print(f"[DEBUG] Full prompt: {prompt}", flush=True)
        except BlockingIOError:
            pass  # Silently skip debug output if it would block
    else:
        # For very large prompts, just log minimal info
        try:
            print(f"[DEBUG] Large prompt: {prompt_len} chars (preview disabled), Max tokens: {max_tokens}", flush=True)
        except BlockingIOError:
            pass
    
    # Add prompt size warning/limit
    MAX_PROMPT_SIZE = 600000  # 600K char hard limit
    if prompt_len > MAX_PROMPT_SIZE:
        return jsonify({"error": f"Prompt too large: {prompt_len} chars (max: {MAX_PROMPT_SIZE}). Please reduce context or history."}), 400
    
    # Log the actual command being run (minimal)
    try:
        print(f"[DEBUG] Running MLX with {prompt_len} char prompt", flush=True)
    except BlockingIOError:
        pass

    try:
        # Ensure mlx venv bin is preferred on PATH for this subprocess
        env = deepcopy(os.environ)
        env['PATH'] = f"{default_mlx_bin}:{env.get('PATH','')}"
        # No timeout - let it run as long as needed
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/Users/Shared', env=env)
        if result.returncode != 0:
            print(f"[ERROR] MLX process failed with return code {result.returncode}")
            print(f"[ERROR] stderr: {result.stderr}")
            print(f"[ERROR] stdout: {result.stdout}")
            return jsonify({"error": result.stderr.strip() or "Kimi K2 process failed"}), 500

        # Check if we got any output at all
        if not result.stdout:
            print(f"[WARNING] MLX returned empty stdout")
            print(f"[WARNING] MLX stderr: {result.stderr}")
            
        # Preserve full model output - no processing, no trimming, no auto-filling
        generated_text = result.stdout
        
        # Strip MLX performance statistics footer if present
        # Format: "==========\nPrompt: X tokens...\nGeneration: Y tokens...\nPeak memory: Z GB\n"
        if "==========" in generated_text:
            # Split at the separator and take only the content before it
            generated_text = generated_text.split("==========")[0].strip()

        # Debug logging with blocking protection
        try:
            output_len = len(generated_text)
            print(f"[DEBUG] Kimi K2 raw output length: {output_len}", flush=True)
            if output_len > 0 and output_len < 10000:
                preview = generated_text[:500] if output_len > 500 else generated_text
                print(f"[DEBUG] Output preview: {preview}", flush=True)
        except BlockingIOError:
            pass  # Skip debug output if it would block
        
        # Return OpenAI-compatible JSON response
        response = {
            "id": "chatcmpl-kimi-k2",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "kimi-k2-local",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        print(f"[DEBUG] Returning JSON response with content length: {len(generated_text)}")
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Expose on all interfaces; adjust port if needed
    app.run(host='0.0.0.0', port=8000)
