1. IBM HARDWARE (ibm_marrakesh, Heron r2, 2026-11-14): bw8 AUC 0.949 [0.917–0.974], vqc32 AUC 0.956 [0.924–0.982] on 200 fold-0 ECGs, 1000 shots. Per-ECG Pearson r 0.93/0.97. AUC survives real hardware. See HARDWARE.md H6.
2. PUSH NEEDED: `git push -u origin smalldata` awaits your approval (all work committed locally on branch smalldata).
3. QB CLOSED: no benefit found. VQC beats LR by +0.006 but loses to tuned LR2 (−0.007) and XGB (−0.014). QB6 did not confirm QB5 candidates. Classified "hypothesis for future work".
4. k=4 SD (secondary endpoint, final): 850/850 complete. All 5 sizes non-inferior; no superiority.
5. Phase 4 redo done: pre-tuned LR2=0.9583, XGB=0.9647 loaded from p1_classical.jsonl.
6. INCOMPLETE: brick-wall at N≠100; torch/TabPFN/IBM blocked by WDAC (use Colab for Qiskit).
