#!/usr/bin/env python3
"""Run existing served benchmarks with public code/prose prompts long enough for 256-token comparisons."""
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_concurrent
import bench_openai

PROMPTS = [
    {'name': 'long-code-chat', 'kind': 'chat', 'prompt':
     'Write a complete Python implementation of a red-black tree with insertion, deletion, searching, '
     'rotations, and extensive unit tests. Include detailed explanations after each method. '
     'Continue until the implementation and every test are complete.'},
    {'name': 'long-prose-chat', 'kind': 'chat', 'prompt':
     'Write a detailed 2000-word tutorial explaining how matrix multiplication runs on a GPU, including '
     'memory layout, tiling, shared memory, synchronization, numerical precision, and several worked '
     'numerical examples. Explain every step thoroughly.'},
]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('decode', 'concurrent'):
        raise SystemExit('Choose decode or concurrent, then pass the existing benchmark client arguments')
    mode = sys.argv.pop(1)
    module = bench_openai if mode == 'decode' else bench_concurrent
    module.PROMPTS = PROMPTS
    original = urllib.request.Request

    def request(url, data=None, *args, **kwargs):
        if data:
            body = json.loads(data)
            body.update(top_p=1, top_k=0, repetition_penalty=1, enable_thinking=False)
            data = json.dumps(body).encode()
        return original(url, data, *args, **kwargs)

    urllib.request.Request = request
    try:
        module.main()
    finally:
        urllib.request.Request = original


if __name__ == '__main__':
    main()
