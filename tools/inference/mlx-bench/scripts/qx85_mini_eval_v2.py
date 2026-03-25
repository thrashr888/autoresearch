#!/usr/bin/env python3
import importlib.util
import json

spec = importlib.util.spec_from_file_location('benchmark_mlx', '/Users/thrashr888/benchmark_mlx.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = 'nightmedia/Qwen3.5-122B-A10B-Text-qx85-mlx'
case_ids = ['reasoning_capital_au', 'reasoning_bat_ball', 'structured_sum_product', 'grounding_unknown']
selected = [c for c in mod.QUALITY_CASES if c['id'] in case_ids]
results = []
for case in selected:
    prompt = mod.build_case_prompt(case)
    attempts = []
    ok = False
    actual = None
    incomplete = False
    for retry_count in (0, 1):
        res = mod.run_generate(model, prompt, mod.quality_max_tokens_for_model(model, case, retry_count=retry_count))
        attempts.append({
            'retry_count': retry_count,
            'gen_tok_s': res.get('gen_tok_s'),
            'peak_memory': res.get('peak_memory'),
            'returncode': res.get('returncode'),
            'response_preview': (res.get('response') or '')[:500],
        })
        if not res['ok']:
            continue
        ok, actual, incomplete = mod.score_case(case, res['response'])
        if ok or not incomplete:
            break
    results.append({
        'id': case['id'],
        'ok': ok,
        'actual': actual,
        'expected': case['expected'],
        'incomplete': incomplete,
        'retried': len(attempts) > 1,
        'attempts': attempts,
    })

score = sum(1 for r in results if r['ok'])
print(json.dumps({'model': model, 'score': score, 'max_score': len(results), 'results': results}, indent=2))
