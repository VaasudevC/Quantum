| model | ECGs | exact (noiseless) | 2000 shots, noiseless | 2000 shots, FakeFez noise | noise + particle-number post-selection | kept shots | compiled CZ / depth (line of FakeFez) |
|---|---|---|---|---|---|---|---|
| brick-wall n=8 (k=4, 4 col) | 300 | 0.915 | 0.909 | 0.911 | 0.911 | 0.65 | 136 / 192 |
| brick-wall n=12 (k=6, 4 col) | 40 | 0.952 | 0.488 | 0.795 | 0.733 | 0.47 | 271 / 287 |

Noiseless shot sweep, **300 stratified fold-0 test ECGs**, sampling from the exact output distribution (`p6_shots.py`):

| model | max abs head weight | exact | 100 | 300 | 1000 | 4000 | 16k | 64k | 256k | 1M |
|---|---|---|---|---|---|---|---|---|---|---|
| n=8 | 4.6 | 0.915 | 0.874 | 0.905 | 0.913 | 0.915 | 0.915 | 0.915 | 0.915 | 0.915 |
| n=12 | 413.7 | 0.954 | 0.542 | 0.532 | 0.535 | 0.609 | 0.624 | 0.719 | 0.858 | 0.939 |

Shot sweep from the Aer statevector simulation on the same 300 (n=8) ECGs: 100 shots: 0.871, 300 shots: 0.895, 1000 shots: 0.911, 4000 shots: 0.912