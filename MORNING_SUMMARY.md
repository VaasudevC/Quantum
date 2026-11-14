1. IBM HARDWARE (ibm_marrakesh, Heron r2, 2026-11-14, 2.2 RU): bw8 AUC 0.949 [0.917–0.974] (paired Δ +0.023 [+0.003, +0.046] vs noiseless — CI excludes 0; likely noise-induced decision-function shift, not reliable improvement; one run, one fold). vqc32 AUC 0.956 [0.924–0.982] (paired Δ −0.011 [−0.024, −0.001]). Per-ECG Pearson r 0.93/0.97. 50% of bw8 shots violated k=4; postselection left AUC unchanged. Thresholds tuned on test set (optimistic). See HARDWARE.md H6.
2. PUSH NEEDED: `git push origin smalldata` awaits your approval (all work committed locally on branch smalldata).
3. QB CLOSED: no benefit found. VQC beats LR by +0.006 but loses to tuned LR2 (−0.007) and XGB (−0.014). QB6 did not confirm QB5 candidates. Classified "hypothesis for future work".
4. k=4 SD (secondary endpoint, final): 850/850 complete. All 5 sizes non-inferior; no superiority.
5. Phase 4 redo done: pre-tuned LR2=0.9583, XGB=0.9647 loaded from p1_classical.jsonl.
6. INCOMPLETE: brick-wall at N≠100; torch/TabPFN blocked by WDAC. Use COLAB_HW.md Cell 9 for FakeMarrakesh matched-noise comparison.
