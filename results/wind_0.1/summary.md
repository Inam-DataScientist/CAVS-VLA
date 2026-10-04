| Method | Train violations | Train success (last 10%) | P(safe landing) | min-start P(safe landing) | P(spec violated) | AG safe (frac. starts) | E[steps] | SMC agrees |
|---|---|---|---|---|---|---|---|---|
| unshielded | 46173.0 ± 222.1 | 0.967 ± 0.002 | 0.9988 ± 0.0006 | 0.8882 ± 0.0317 | 1.23e-03 ± 6.03e-04 | 0.116 ± 0.042 | 8.09 ± 0.22 | 1.00 ± 0.00 |
| prob_shield | 399.8 ± 32.1 | 1.000 ± 0.000 | 0.9999 ± 0.0001 | 0.9882 ± 0.0099 | 1.26e-04 ± 1.16e-04 | 0.126 ± 0.032 | 9.27 ± 0.37 | 1.00 ± 0.00 |
| sure_shield | 0.0 ± 0.0 | 1.000 ± 0.000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.90e-12 ± 0.00e+00 | 0.847 ± 0.000 | 8.74 ± 0.13 | 1.00 ± 0.00 |

mean ± 95% t-CI over 5 seeds; probabilities are exact (PCTL), averaged over the uniform start distribution.

Shield vs. learning (exact): learned greedy policy vs. uniform-random actions inside the same shield

| Shield | learned P(safe landing) | random P(safe landing) | learned E[steps] | random E[steps] |
|---|---|---|---|---|
| unshielded | 0.9988 | 0.0029 | 8.09 | 6.61 |
| prob_shield | 0.9999 | 0.9915 | 9.27 | 21.25 |
| sure_shield | 1.0000 | 1.0000 | 8.74 | 21.48 |
