#!/bin/sh
# usage: qb4_run.sh FOLD  -- trains the pre-registered placements of one fold sequentially (resumable)
F=$1
NAMES=$(../.venv/Scripts/python.exe - "$F" <<'PYEND'
import json, sys
f=int(sys.argv[1]); R=[json.loads(l) for l in open('results/qb4_screen.jsonl') if json.loads(l)['fold']==f]
b=sorted([(r['calign'],r['name']) for r in R if r['name'].startswith('b_rand')])
names=['a_L1','c_paired','d_best_of_200']+[n for _,n in b[-2:]]+[n for _,n in b[:2]]
done=set()
try:
    for l in open('results/qb4_train.jsonl'):
        r=json.loads(l); done.add((r['fold'],r['name']))
except FileNotFoundError: pass
print(' '.join(n for n in names if (f,n) not in done))
PYEND
)
for n in $NAMES; do OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/Scripts/python.exe -u qb4_placement.py train $F $n; done
