| Method | Train violations | Train success (last 10%) | P(safe landing) | min-start P(safe landing) | P(spec violated) | AG safe (frac. starts) | E[steps] | SMC agrees |
|---|---|---|---|---|---|---|---|---|
| unshielded | 48588.2 ± 192.5 | 0.956 ± 0.002 | 0.9995 ± 0.0003 | 0.8490 ± 0.2055 | 4.87e-04 ± 3.31e-04 | 0.170 ± 0.039 | 8.34 ± 0.10 | 1.00 ± 0.00 |
| prob_shield | 701.2 ± 26.2 | 0.999 ± 0.000 | 0.9999 ± 0.0000 | 0.9967 ± 0.0040 | 9.75e-05 ± 2.77e-05 | 0.154 ± 0.022 | 9.34 ± 0.16 | 1.00 ± 0.00 |
| sure_shield | 0.0 ± 0.0 | 1.000 ± 0.000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.23e-09 ± 0.00e+00 | 0.847 ± 0.000 | 9.01 ± 0.09 | 1.00 ± 0.00 |

mean ± 95% t-CI over 10 seeds; probabilities are exact (PCTL), averaged over the uniform start distribution.

Shield vs. learning (exact): learned greedy policy vs. uniform-random actions inside the same shield

| Shield | learned P(safe landing) | random P(safe landing) | learned E[steps] | random E[steps] |
|---|---|---|---|---|
| unshielded | 0.9995 | 0.0026 | 8.34 | 6.07 |
| prob_shield | 0.9999 | 0.9826 | 9.34 | 21.27 |
| sure_shield | 1.0000 | 1.0000 | 9.01 | 21.50 |
