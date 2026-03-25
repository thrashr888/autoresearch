#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time

SPEED_PROMPT = 'Explain recursion to a smart teenager in about 180 words. No bullets.'

QUALITY_CASES = [
    {
        'id': 'reasoning_capital_au',
        'category': 'reasoning',
        'prompt': 'Return ONLY one word: capital of Australia.',
        'expected': 'canberra',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'reasoning_sheep',
        'category': 'reasoning',
        'prompt': 'Return ONLY digits: A farmer has 17 sheep and all but 9 die. How many are left?',
        'expected': '9',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'reasoning_bat_ball',
        'category': 'reasoning',
        'prompt': 'Return ONLY a decimal number: A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?',
        'expected': '0.05',
        'answer_type': 'scalar',
        'max_tokens': 32,
    },
    {
        'id': 'reasoning_arithmetic',
        'category': 'reasoning',
        'prompt': 'Return ONLY digits: what is 1729 + 431?',
        'expected': '2160',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'reasoning_machines',
        'category': 'reasoning',
        'prompt': 'Return ONLY digits: If 5 machines make 5 widgets in 5 minutes, how many minutes would 100 machines take to make 100 widgets?',
        'expected': '5',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'reasoning_weekday',
        'category': 'reasoning',
        'prompt': 'Return ONLY one word: if today is Tuesday, what day is 10 days from now?',
        'expected': 'friday',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'structured_sorted_unique',
        'category': 'structured',
        'prompt': 'Return ONLY a minified JSON array: sorted unique integers from [3,1,3,2].',
        'expected': [1, 2, 3],
        'answer_type': 'json',
        'max_tokens': 40,
    },
    {
        'id': 'structured_sum_product',
        'category': 'structured',
        'prompt': 'Return ONLY minified JSON with keys sum and product for the numbers 7 and 8.',
        'expected': {'sum': 15, 'product': 56},
        'answer_type': 'json',
        'max_tokens': 48,
    },
    {
        'id': 'structured_even_filter',
        'category': 'structured',
        'prompt': 'Return ONLY a minified JSON array of the sorted unique even numbers from [5,2,8,2,3,10].',
        'expected': [2, 8, 10],
        'answer_type': 'json',
        'max_tokens': 40,
    },
    {
        'id': 'structured_slug',
        'category': 'structured',
        'prompt': 'Return ONLY the lowercase slug for this title, using hyphens: MLX Benchmark Suite',
        'expected': 'mlx-benchmark-suite',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'extraction_email_domain',
        'category': 'extraction',
        'prompt': 'Return ONLY the email domain from paul@example.com',
        'expected': 'example.com',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'extraction_json_key_count',
        'category': 'extraction',
        'prompt': 'Return ONLY one digit: how many top-level keys are in this JSON object: {"a":1,"b":2,"c":3}?',
        'expected': '3',
        'answer_type': 'scalar',
        'max_tokens': 16,
    },
    {
        'id': 'extraction_json_math',
        'category': 'extraction',
        'prompt': 'Return ONLY digits: given JSON {"a":3,"b":5}, what is a+b?',
        'expected': '8',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'extraction_csv_unique',
        'category': 'extraction',
        'prompt': 'Return ONLY comma-separated values with no spaces: unique sorted words from apple,banana,apple,cherry',
        'expected': 'apple,banana,cherry',
        'answer_type': 'scalar',
        'max_tokens': 32,
    },
    {
        'id': 'grounding_fact_lookup',
        'category': 'grounding',
        'prompt': 'Facts: project codename is Argos. Question: what is the project codename? Return ONLY the answer.',
        'expected': 'argos',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'grounding_unknown',
        'category': 'grounding',
        'prompt': 'Facts: project codename is Argos. The facts do not mention the release date. Question: what is the release date? If unknown, return ONLY unknown.',
        'expected': 'unknown',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'grounding_contradiction',
        'category': 'grounding',
        'prompt': 'Facts: the only supported colors are red and blue. Question: is green supported? Return ONLY yes or no.',
        'expected': 'no',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
    {
        'id': 'grounding_count_lines',
        'category': 'grounding',
        'prompt': 'Data: line1=alpha; line2=beta; line3=gamma. Return ONLY digits: how many lines are listed?',
        'expected': '3',
        'answer_type': 'scalar',
        'max_tokens': 24,
    },
]


def should_ignore_chat_template(model):
    name = (model or '').lower()
    return 'qwen3.5' in name and 'text' in name


def build_case_prompt(case):
    if case['answer_type'] == 'json':
        suffix = 'If you think, keep it brief. End with exactly one line starting with FINAL: followed by valid minified JSON only.'
    else:
        suffix = 'If you think, keep it brief. End with exactly one line starting with FINAL: followed by only the answer.'
    return f"{case['prompt']}\n\n{suffix}"


def build_generate_cmd(model, prompt, max_tokens):
    cmd = [
        sys.executable, '-m', 'mlx_lm.generate',
        '--model', model,
        '--prompt', prompt,
        '--max-tokens', str(max_tokens),
        '--temp', '0',
    ]
    if should_ignore_chat_template(model):
        cmd.append('--ignore-chat-template')
    return cmd


def run_generate(model, prompt, max_tokens):
    cmd = build_generate_cmd(model, prompt, max_tokens)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - started, 3)
    stdout = proc.stdout or ''
    stderr = proc.stderr or ''
    merged = (stdout + '\n' + stderr).strip()
    response = ''
    m = re.search(r'=+\n(.*?)\n=+\nPrompt:', stdout, re.S)
    if m:
        response = m.group(1).strip()
    prompt_stats = re.search(r'Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec', merged)
    gen_stats = re.search(r'Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec', merged)
    peak_mem = re.search(r'Peak memory:\s+([^\n]+)', merged)
    return {
        'ok': proc.returncode == 0,
        'returncode': proc.returncode,
        'elapsed_s': elapsed,
        'response': response,
        'prompt_tokens': int(prompt_stats.group(1)) if prompt_stats else None,
        'prompt_tok_s': float(prompt_stats.group(2)) if prompt_stats else None,
        'gen_tokens': int(gen_stats.group(1)) if gen_stats else None,
        'gen_tok_s': float(gen_stats.group(2)) if gen_stats else None,
        'peak_memory': peak_mem.group(1).strip() if peak_mem else None,
        'stdout_tail': stdout[-2000:],
        'stderr_tail': stderr[-2000:],
    }


def extract_final_response(text):
    text = (text or '').strip()
    if not text:
        return text

    tagged_final = re.search(
        r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)',
        text,
        re.S,
    )
    if tagged_final:
        return tagged_final.group(1).strip()

    channel_final = re.search(r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)', text, re.S)
    if channel_final:
        return channel_final.group(1).strip()

    return text


def clean_response(text):
    text = extract_final_response(text)
    final_line = re.search(r'(?im)^FINAL:\s*(.+)$', text)
    if final_line:
        return final_line.group(1).strip()

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    text = re.sub(r'(?im)^thinking process:\s*', '', text)
    text = re.sub(r'(?im)^return only minified json\.?\s*', '', text)
    text = text.strip()

    fenced = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
    if fenced:
        text = fenced.group(1).strip()

    return text.strip()


def extract_json_candidates(text):
    text = text or ''
    decoder = json.JSONDecoder()
    out = []
    for i, ch in enumerate(text):
        if ch not in '{[':
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except Exception:
            continue
        out.append(obj)
    return out


def extract_json_like(text):
    raw_text = (text or '').strip()
    text = clean_response(raw_text)
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    for candidate_source in (text, raw_text):
        candidates = extract_json_candidates(candidate_source)
        if candidates:
            return candidates[-1]

    return None


def norm_scalar(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            return ('%.10f' % v).rstrip('0').rstrip('.')
        return str(v)
    text = str(v).strip().lower()
    text = text.strip(" \t\n\r.,;:!?\"'`@")
    return text


def extract_scalar_answer(text):
    text = clean_response(text)
    if not text:
        return ''

    json_like = extract_json_like(text)
    if isinstance(json_like, (str, int, float)):
        return str(json_like).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.strip()
    return lines[-1]


def answers_match(actual, expected):
    if isinstance(expected, (dict, list)):
        return actual == expected
    return norm_scalar(actual) == norm_scalar(expected)


def response_looks_incomplete(case, raw_text, actual):
    text = (raw_text or '').strip()
    cleaned = clean_response(raw_text)
    lowered = text.lower()
    has_final = re.search(r'(?im)^FINAL:\s*(.+)$', text) is not None

    if has_final:
        if case['answer_type'] == 'json' and isinstance(actual, (dict, list)):
            return False
        if case['answer_type'] == 'scalar' and norm_scalar(actual):
            return False

    incomplete_markers = [
        'thinking process:',
        '<|channel|>analysis',
        '<think>',
        'step 1:',
        'analyze the request:',
        'task:',
        'input:',
        'goal:',
    ]
    if (not has_final) and any(marker in lowered for marker in incomplete_markers):
        return True

    if case['answer_type'] == 'json':
        opens = text.count('{') + text.count('[')
        closes = text.count('}') + text.count(']')
        if opens > closes:
            return True

    if case['answer_type'] == 'scalar' and not has_final:
        if cleaned.startswith('*') or cleaned.startswith('-') or cleaned.startswith('1.'):
            return True
        if len(cleaned.split()) > 6:
            return True

    return False


def score_case(case, raw_text):
    if case['answer_type'] == 'json':
        actual = extract_json_like(raw_text)
    else:
        actual = extract_scalar_answer(raw_text)
    ok = answers_match(actual, case['expected'])
    incomplete = (not ok) and response_looks_incomplete(case, raw_text, actual)
    return ok, actual, incomplete


def quality_max_tokens_for_model(model, case, retry_count=0):
    name = (model or '').lower()
    base = case.get('max_tokens', 32)
    if 'gpt-oss' in name:
        initial = max(base * 4, 96)
        return initial if retry_count == 0 else max(initial * 2, 192)
    if 'qwen3.5' in name and 'text' in name:
        initial = base
        return initial if retry_count == 0 else max(initial * 2, 96)
    initial = max(base * 2, 48)
    return initial if retry_count == 0 else max(initial * 2, 96)


def run_quality_suite(model):
    details = []
    total_score = 0
    gen_speeds = []
    peak_memories = []
    by_category = {}

    for idx, case in enumerate(QUALITY_CASES, 1):
        prompt = build_case_prompt(case)
        attempts = []
        ok = False
        actual = None
        incomplete = False

        for retry_count in (0, 1):
            res = run_generate(model, prompt, quality_max_tokens_for_model(model, case, retry_count=retry_count))
            attempts.append(res)
            if not res['ok']:
                continue
            ok, actual, incomplete = score_case(case, res['response'])
            if ok or not incomplete:
                break

        item = {
            'id': case['id'],
            'category': case['category'],
            'prompt': case['prompt'],
            'expected': case['expected'],
            'attempts': attempts,
            'result': attempts[-1] if attempts else None,
            'actual': actual,
            'correct': ok,
            'incomplete': incomplete,
            'retried': len(attempts) > 1,
        }

        final_res = item['result'] or {}
        if final_res.get('ok'):
            total_score += int(ok)
            if final_res.get('gen_tok_s') is not None:
                gen_speeds.append(final_res['gen_tok_s'])
            if final_res.get('peak_memory'):
                peak_memories.append(final_res['peak_memory'])
        else:
            item['error'] = final_res.get('stderr_tail') or final_res.get('stdout_tail')

        bucket = by_category.setdefault(case['category'], {'score': 0, 'max': 0, 'incomplete': 0})
        bucket['score'] += int(item['correct'])
        bucket['max'] += 1
        bucket['incomplete'] += int(item.get('incomplete', False))
        details.append(item)
        status = 'ok' if item['correct'] else ('incomplete' if item.get('incomplete') else 'miss')
        retry_note = ' retry' if item.get('retried') else ''
        print(f"  quality case {idx}/{len(QUALITY_CASES)} {case['id']}: {status}{retry_note}", flush=True)

    return {
        'score': total_score,
        'max_score': len(QUALITY_CASES),
        'details': details,
        'by_category': by_category,
        'avg_gen_tok_s': round(statistics.mean(gen_speeds), 3) if gen_speeds else None,
        'peak_memory_observed': peak_memories[-1] if peak_memories else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True, help='Local model path or HF repo; repeat for multiple models')
    ap.add_argument('--out-json', default='/Users/thrashr888/mlx_bench_results.json')
    ap.add_argument('--out-csv', default='/Users/thrashr888/mlx_bench_results.csv')
    args = ap.parse_args()

    results = []
    for idx, model in enumerate(args.model, 1):
        item = {'model': model, 'started_at': time.time()}
        print(f'[{idx}/{len(args.model)}] {model}', flush=True)

        q = run_quality_suite(model)
        item['quality'] = q
        item['quality_score'] = q['score']
        item['quality_max'] = q['max_score']
        print(f"  quality total {q['score']}/{q['max_score']} avg {q['avg_gen_tok_s']} tok/s peak {q['peak_memory_observed']}", flush=True)

        s = run_generate(model, SPEED_PROMPT, 220)
        item['speed'] = s
        if s['ok']:
            print(f"  speed {s['gen_tok_s']} tok/s peak {s['peak_memory']}", flush=True)
        else:
            item['speed_error'] = s['stderr_tail'] or s['stdout_tail']
            print(f"  speed error rc={s['returncode']}", flush=True)

        item['finished_at'] = time.time()
        item['elapsed_s'] = round(item['finished_at'] - item['started_at'], 2)
        results.append(item)

        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)

        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'model', 'quality_score', 'quality_max', 'quality_pct',
                'quality_avg_gen_tok_s', 'speed_gen_tok_s', 'speed_elapsed_s', 'peak_memory'
            ])
            for r in results:
                q = r.get('quality', {})
                s = r.get('speed', {})
                max_score = q.get('max_score') or 0
                pct = round((100.0 * q.get('score', 0) / max_score), 1) if max_score else None
                w.writerow([
                    r['model'],
                    q.get('score'),
                    max_score,
                    pct,
                    q.get('avg_gen_tok_s'),
                    s.get('gen_tok_s'),
                    s.get('elapsed_s'),
                    s.get('peak_memory') or q.get('peak_memory_observed'),
                ])

    print(args.out_json)
    print(args.out_csv)


if __name__ == '__main__':
    main()
