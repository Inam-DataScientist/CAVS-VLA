| Method | Train violations | Train success (last 10%) | P(safe landing) | min-start P(safe landing) | P(spec violated) | AG safe (frac. starts) | E[steps] | SMC agrees |
|---|---|---|---|---|---|---|---|---|
| unshielded | 53518.2 ± 393.6 | 0.925 ± 0.001 | 0.9978 ± 0.0003 | 0.6070 ± 0.2349 | 2.24e-03 ± 3.34e-04 | 0.359 ± 0.034 | 8.88 ± 0.15 | 1.00 ± 0.00 |
| prob_shield | 623.2 ± 42.3 | 0.999 ± 0.000 | 0.9997 ± 0.0001 | 0.9954 ± 0.0017 | 2.76e-04 ± 6.61e-05 | 0.197 ± 0.013 | 10.47 ± 0.17 | 1.00 ± 0.00 |
| sure_shield | 0.0 ± 0.0 | 1.000 ± 0.000 | 1.0000 ± 0.0000 | 0.9997 ± 0.0000 | 1.14e-06 ± 0.00e+00 | 0.847 ± 0.000 | 8.86 ± 0.04 | 1.00 ± 0.00 |

mean ± 95% t-CI over 5 seeds; probabilities are exact (PCTL), averaged over the uniform start distribution.

Shield vs. learning (exact): learned greedy policy vs. uniform-random actions inside the same shield

| Shield | learned P(safe landing) | random P(safe landing) | learned E[steps] | random E[steps] |
|---|---|---|---|---|
| unshielded | 0.9978 | 0.0021 | 8.88 | 5.10 |
| prob_shield | 0.9997 | 0.9854 | 10.47 | 21.45 |
| sure_shield | 1.0000 | 1.0000 | 8.86 | 21.57 |
