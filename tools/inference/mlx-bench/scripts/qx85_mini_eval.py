#!/usr/bin/env python3
import json
import re
import statistics
import subprocess
import sys

PY = '/Users/thrashr888/.venvs/mlx-bench-qwen/bin/python'
MODEL = 'nightmedia/Qwen3.5-122B-A10B-Text-qx85-mlx'
CASES = [
    ('reasoning_capital_au', 'Return ONLY one word: capital of Australia.', 'canberra'),
    ('reasoning_bat_ball', 'Return ONLY a decimal number: A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?', '0.05'),
    ('structured_sum_product', 'Return ONLY minified JSON with keys sum and product for the numbers 7 and 8.', {'sum': 15, 'product': 56}),
    ('grounding_unknown', 'Facts: project codename is Argos. The facts do not mention the release date. Question: what is the release date? If unknown, return ONLY unknown.', 'unknown'),
]

def clean(text):
    text = (text or '').strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    text = re.sub(r'(?im)^return only minified json\.?\s*', '', text)
    return text.strip()

def norm(v):
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            return ('%.10f' % v).rstrip('0').rstrip('.')
        return str(v)
    return str(v).strip().lower().strip(" \t\n\r.,;:!?\"'`@")

def parse_response(stdout):
    m = re.search(r'=+\n(.*?)\n=+\nPrompt:', stdout, re.S)
    return clean(m.group(1).strip() if m else '')

results = []
for cid, prompt, expected in CASES:
    cmd = [PY, '-m', 'mlx_lm.generate', '--model', MODEL, '--ignore-chat-template', '--prompt', prompt, '--max-tokens', '96', '--temp', '0']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout or ''
    merged = (proc.stdout or '') + '\n' + (proc.stderr or '')
    response = parse_response(out)
    gen = re.search(r'Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec', merged)
    peak = re.search(r'Peak memory:\s+([^\n]+)', merged)
    actual = response
    ok = False
    if isinstance(expected, dict):
        try:
            actual = json.loads(response)
        except Exception:
            actual = response
        ok = actual == expected
    else:
        ok = norm(actual) == norm(expected)
    results.append({
        'id': cid,
        'ok': ok,
        'expected': expected,
        'actual': actual,
        'gen_tok_s': float(gen.group(2)) if gen else None,
        'peak_memory': peak.group(1).strip() if peak else None,
        'returncode': proc.returncode,
    })

score = sum(1 for r in results if r['ok'])
print(json.dumps({
    'model': MODEL,
    'score': score,
    'max_score': len(results),
    'avg_gen_tok_s': round(statistics.mean([r['gen_tok_s'] for r in results if r['gen_tok_s'] is not None]), 3),
    'peak_memory_last': next((r['peak_memory'] for r in reversed(results) if r['peak_memory']), None),
    'results': results,
}, indent=2))
