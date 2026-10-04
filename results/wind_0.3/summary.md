| Method | Train violations | Train success (last 10%) | P(safe landing) | min-start P(safe landing) | P(spec violated) | AG safe (frac. starts) | E[steps] | SMC agrees |
|---|---|---|---|---|---|---|---|---|
| unshielded | 50973.6 ± 203.9 | 0.944 ± 0.004 | 0.9993 ± 0.0004 | 0.8091 ± 0.2856 | 6.88e-04 ± 4.47e-04 | 0.248 ± 0.035 | 8.62 ± 0.22 | 1.00 ± 0.00 |
| prob_shield | 549.4 ± 35.0 | 1.000 ± 0.000 | 0.9998 ± 0.0001 | 0.9973 ± 0.0015 | 1.61e-04 ± 1.25e-04 | 0.215 ± 0.036 | 9.93 ± 0.37 | 1.00 ± 0.00 |
| sure_shield | 0.0 ± 0.0 | 1.000 ± 0.000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 6.23e-08 ± 0.00e+00 | 0.847 ± 0.000 | 8.92 ± 0.13 | 1.00 ± 0.00 |

mean ± 95% t-CI over 5 seeds; probabilities are exact (PCTL), averaged over the uniform start distribution.

Shield vs. learning (exact): learned greedy policy vs. uniform-random actions inside the same shield

| Shield | learned P(safe landing) | random P(safe landing) | learned E[steps] | random E[steps] |
|---|---|---|---|---|
| unshielded | 0.9993 | 0.0023 | 8.62 | 5.56 |
| prob_shield | 0.9998 | 0.9867 | 9.93 | 21.36 |
| sure_shield | 1.0000 | 1.0000 | 8.92 | 21.54 |
