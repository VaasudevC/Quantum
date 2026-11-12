"""EXTRA 4: brick-wall at 16 qubits (16 features, k=4 -> sector dimension C(16,4)=1820; k=6 would be 8008 and ~10 s/step, infeasible here).
Config chosen a priori from the n=8/12 results (scale pi/4, 4 columns, linear head, default particle number max(2, 2*(n//8)) = 4); 5 folds, 1500 steps (+extension).
-> results/bw/extra4_n16.jsonl"""
import sys
import p3_brickwall as P

def main():
    cfgs = [P.base(n=16, k=4, columns=4, scale=0.7853981633974483, fold=f, save_scores=True) for f in range(5)]
    P.runner(cfgs, "results/bw/extra4_n16.jsonl", int(sys.argv[1]) if len(sys.argv) > 1 else 3)

if __name__ == "__main__":
    main()
