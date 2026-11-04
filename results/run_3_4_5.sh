#!/bin/bash
cd "$(dirname "$0")/.."
python -u scale_cv.py --scales "pi/6" --ks 8,16,32 --selectors l1,cohen_dedup,random --seeds 0,1,2 --steps 1000 --jobs 3 --out results/stage3_selectors.jsonl > results/stage3.log 2>&1
python -u scale_cv.py --scales "pi/6,pi" --ks 16,32 --steps 1000 --jobs 3 --out results/stage4_real.jsonl > results/stage4_real.log 2>&1
python -u scale_cv.py --scales "pi/6,pi" --ks 16,32 --steps 1000 --jobs 3 --shuffle-later --out results/stage4_shuffled.jsonl > results/stage4_shuffled.log 2>&1
python -u scale_cv.py --split official --scales "pi/6,pi/2,pi" --ks 8,32 --steps 1000 --jobs 3 --out results/stage5_official.jsonl > results/stage5.log 2>&1
echo ALLDONE > results/chain.done
