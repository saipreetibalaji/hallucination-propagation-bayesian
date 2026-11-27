"""Step 1 (generation) as a script for Mac / any machine.

Usage:
    .venv/bin/python run_generation.py --n-chains 3 --out test.jsonl     # quick test
    .venv/bin/python run_generation.py --n-chains 300 --out generations.jsonl

Resumable: re-running skips chains already written to --out.
"""
import argparse, json, os, random, re, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--n-chains', type=int, default=300)
ap.add_argument('--min-hops', type=int, default=3)
ap.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
ap.add_argument('--out', default='generations.jsonl')
ap.add_argument('--no-paragraph', action='store_true', help='closed-book (no supporting paragraph)')
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

ds = load_dataset('dgslibisey/MuSiQue', split='validation')
chains = [ex for ex in ds if ex['answerable'] and len(ex['question_decomposition']) >= args.min_hops]
random.Random(args.seed).shuffle(chains)
chains = chains[:args.n_chains]
print('chains:', len(chains), flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
model.eval()

SYSTEM = ("Answer the question with a short phrase only (a name, place, date or number). "
          "If you do not know, answer exactly: unsure.")


def to_question(q):
    """MuSiQue sub-questions are sometimes 'subject >> relation'; make those natural."""
    if '>>' in q:
        subj, rel = [s.strip() for s in q.split('>>', 1)]
        return f"What is the '{rel}' of '{subj}'?"
    return q


def build_messages(history, question, paragraph=None):
    msgs = [{'role': 'system', 'content': SYSTEM}]
    for q, a in history:
        msgs.append({'role': 'user', 'content': q})
        msgs.append({'role': 'assistant', 'content': a})
    u = question if not paragraph else f"Context: {paragraph}\n\nQuestion: {question}"
    msgs.append({'role': 'user', 'content': u})
    return msgs


@torch.no_grad()
def ask(msgs, max_new_tokens=24):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors='pt').to(device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         output_scores=True, return_dict_in_generate=True,
                         pad_token_id=tok.eos_token_id)
    gen = out.sequences[0, inp['input_ids'].shape[1]:]
    probs = [torch.softmax(s[0].float(), -1)[gen[i]].item() for i, s in enumerate(out.scores)]
    ans = tok.decode(gen, skip_special_tokens=True).strip()
    return ans, (float(np.mean(probs)) if probs else 0.0), (float(np.min(probs)) if probs else 0.0)


def run_chain(ex, mode):
    dec, paras = ex['question_decomposition'], ex['paragraphs']
    gold = [d['answer'] for d in dec]
    model_ans, history, steps = [], [], []
    for k, d in enumerate(dec):
        src = gold if mode == 'independent' else model_ans
        q = d['question']
        for j in range(k):
            q = q.replace(f'#{j+1}', src[j])
        q = to_question(q)
        para = None
        if not args.no_paragraph:
            idx = d['paragraph_support_idx']
            if idx is not None and 0 <= idx < len(paras):
                para = paras[idx]['paragraph_text']
        msgs = build_messages(history if mode == 'dependent' else [], q, para)
        ans, p_mean, p_min = ask(msgs)
        model_ans.append(ans)
        if mode == 'dependent':
            history.append((q, ans))
        steps.append(dict(step=k + 1, question=q, gold=gold[k], pred=ans, p_mean=p_mean, p_min=p_min))
    return dict(id=ex['id'], mode=mode, steps=steps)


done = set()
if os.path.exists(args.out):
    with open(args.out) as f:
        for line in f:
            r = json.loads(line); done.add((r['id'], r['mode']))
print('already done:', len(done), flush=True)

t0 = time.time(); n = 0
with open(args.out, 'a') as f:
    for i, ex in enumerate(chains):
        for mode in ('independent', 'dependent'):
            if (ex['id'], mode) in done:
                continue
            f.write(json.dumps(run_chain(ex, mode)) + '\n'); f.flush()
            n += 1
        el = time.time() - t0
        print(f'chain {i+1}/{len(chains)}  elapsed {el/60:.1f} min  avg {el/max(n,1):.1f}s per chain-run', flush=True)
print('finished', flush=True)
