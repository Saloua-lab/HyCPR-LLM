"""Parse results_*.txt logs into runs of (true, predicted) rating pairs.

Applies the evaluation notebook's rule: a pair counts only if both ratings parse
and the prediction is within 1..5 (NewEvaluation18Nov_Updated.ipynb cell 44).
"""
import re, sys, json, glob, os, math

def num(x):
    try: return float(x)
    except ValueError: return None

def parse(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    header = txt.split('\n', 1)[0]
    runs, cur, last = [], [], None
    for it, t, p in re.findall(r'Iteration:\s*(\d+)\s*\nTrue Rating:\s*(\S+)\s*\nPredicted Rating:\s*(\S+)', txt):
        it = int(it)
        if last is not None and it <= last:
            runs.append(cur); cur = []
        cur.append((num(t), num(p))); last = it
    if cur: runs.append(cur)
    return header, runs

def main(folder, out):
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, '**', 'results_*.txt'), recursive=True)):
        if '.ipynb_checkpoints' in path: continue
        header, runs = parse(path)
        for k, run in enumerate(runs):
            valid = [(t, p) for t, p in run if t is not None and p is not None and 1 <= t <= 5 and 1 <= p <= 5]
            mae = rmse = None
            if valid:
                mae = sum(abs(t - p) for t, p in valid) / len(valid)
                rmse = math.sqrt(sum((t - p) ** 2 for t, p in valid) / len(valid))
            rows.append(dict(file=os.path.relpath(path, folder), header=header, run=k, n=len(run),
                             valid=len(valid), mae=mae, rmse=rmse, pairs=valid))
    json.dump(rows, open(out, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
