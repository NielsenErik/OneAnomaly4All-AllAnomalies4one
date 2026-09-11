# Diagnosis report — diagnosis_controls

EXPLORATORY. Generated from completed validation-split runs. No PASS/FAIL, no confirmatory claim, and no endpoint here was frozen before the data were seen.

## Run inventory

- runs found: 30
- usable (run ok and diagnosis ok): 30
- refused: 0

- result rows used: 2220 of 2220 (0 dropped by provenance gating)

## Models and optimisation

| variant | method | parameters_mean | parameters_std | nodes_mean | nodes_std | fit_s_mean | fit_s_std | optimizer_steps_mean | optimizer_steps_std | epochs_run_mean | epochs_run_std | best_epoch_mean | best_epoch_std | checkpoint_nll_mean | checkpoint_nll_std | final_train_nll_mean | final_train_nll_std | log_partition_mean | log_partition_std | lr_mean | lr_std | epochs_budget_mean | epochs_budget_std | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocked | model | 1488 | 0 | 361 | 0 | 2.536 | 0.6972 | 1061 | 169.8 | 132.7 | 21.22 | 111.7 | 28.02 | 19.62 | 0.3415 | 19.53 | 0.3643 | -3.974e-08 | 1.241e-07 | 0.02 | 0 | 150 | 0 | 3 |
| conditional | model | 1488 | 0 | 361 | 0 | 9.833 | 1.253 | 1077 | 143.2 | 134.7 | 17.9 | 113.3 | 24.5 | 19.28 | 0.3816 | 19.11 | 0.2466 | 1.987e-08 | 1.241e-07 | 0.02 | 0 | 150 | 0 | 3 |
| conditional_nll_selection | model | 1488 | 0 | 361 | 0 | 8.051 | 0.9421 | 1077 | 143.2 | 134.7 | 17.9 | 113.3 | 24.5 | 19.28 | 0.3816 | 19.11 | 0.2466 | 1.987e-08 | 1.241e-07 | 0.02 | 0 | 150 | 0 | 3 |
| interface | model | 2688 | 0 | 529 | 0 | 3.055 | 0.8403 | 1045 | 267.9 | 130.7 | 33.49 | 111 | 39.36 | 19.24 | 0.2126 | 19.06 | 0.02976 | -1.788e-07 | 1.788e-07 | 0.02 | 0 | 150 | 0 | 3 |
| interface_conditional | model | 2688 | 0 | 529 | 0 | 12.55 | 2.262 | 1011 | 180.7 | 126.3 | 22.59 | 108.7 | 36.2 | 19.04 | 0.1495 | 18.94 | 0.05288 | 5.96e-08 | 1.577e-07 | 0.02 | 0 | 150 | 0 | 3 |
| mixture | model | 1608 | 0 | 345 | 0 | 2.27 | 0.7586 | 1021 | 309.5 | 127.7 | 38.68 | 111.3 | 47.65 | 18.13 | 0.4324 | 17.91 | 0.04051 | 0 | 1.192e-07 | 0.02 | 0 | 150 | 0 | 3 |
| random | model | 1488 | 0 | 361 | 0 | 3.269 | 0.17 | 1168 | 55.43 | 146 | 6.928 | 133.3 | 19.14 | 19.37 | 0.6845 | 19.12 | 0.4195 | -1.987e-08 | 9.105e-08 | 0.02 | 0 | 150 | 0 | 3 |
| unblocked | model | 1488 | 0 | 361 | 0 | 3.422 | 0.4315 | 1117 | 143.2 | 139.7 | 17.9 | 130.3 | 32.33 | 16.5 | 0.5114 | 16.38 | 0.07497 | 1.987e-07 | 3.283e-07 | 0.02 | 0 | 150 | 0 | 3 |
| uniform_wide | model | 8384 | 0 | 1201 | 0 | 5.997 | 0.4833 | 858.7 | 86.29 | 107.3 | 10.79 | 81.33 | 10.79 | 19.08 | 0.1375 | 18.52 | 0.2509 | 3.974e-08 | 6.883e-08 | 0.02 | 0 | 150 | 0 | 3 |
| within_capacity | model | 6416 | 0 | 1033 | 0 | 6.048 | 0.5561 | 1061 | 121.2 | 132.7 | 15.14 | 114.7 | 28.94 | 19.69 | 0.45 | 19.06 | 0.3349 | 8.925e-08 | 3.113e-07 | 0.02 | 0 | 150 | 0 | 3 |

## Detection

| variant | method_name | mask | score | auroc_mean | auroc_std | ap_mean | ap_std | fpr_mean | fpr_std | power_mean | power_std | score_range_mean | score_range_std | n_seeds | numerically_flat_frac | threshold_infinite_frac | depends_on_values_frac | depends_on_fault_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocked | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.06167 | 0.03753 | 0.06167 | 0.03753 | 14.07 | 4.669 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.0401 | 0.05833 | 0.0401 | 23.66 | 5.58 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.06167 | 0.04252 | 0.06167 | 0.04252 | 23.35 | 5.375 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.05 | 0.02291 | 0.05 | 0.02291 | 14.26 | 4.822 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.03667 | 0.01443 | 0.03667 | 0.01443 | 1.585 | 1.376 | 3 | 0.3333 | 0 | 0 | 0 |
| blocked | circuit | full | conditional_max | 0.6448 | 0.006925 | 0.6457 | 0.009397 | 0.055 | 0.04444 | 0.15 | 0.06557 | 15.47 | 1.881 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | independent | 0.5232 | 0.001538 | 0.4972 | 0.0003352 | 0.05333 | 0.02309 | 0.03333 | 0.02309 | 45.19 | 11.45 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | joint | 0.5881 | 0.004702 | 0.5513 | 0.002922 | 0.05333 | 0.04646 | 0.06333 | 0.0562 | 44.01 | 11.35 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | marginal_max | 0.5632 | 0.00514 | 0.5433 | 0.001056 | 0.04833 | 0.03617 | 0.05667 | 0.04726 | 16.37 | 1.153 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | relational_max | 0.7103 | 0.01162 | 0.7459 | 0.01318 | 0.05667 | 0.01443 | 0.3217 | 0.03884 | 6.263 | 1.092 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.04856 | 0.05667 | 0.04856 | 13.91 | 4.57 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.04833 | 0.03617 | 0.04833 | 0.03617 | 35.36 | 11.02 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.05333 | 0.04072 | 0.05333 | 0.04072 | 34.69 | 10.57 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.045 | 0.03606 | 0.045 | 0.03606 | 14.23 | 4.829 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06333 | 0.01893 | 0.06333 | 0.01893 | 2.285 | 2.059 | 3 | 0.3333 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | conditional_max | 0.6503 | 0.02494 | 0.6505 | 0.02985 | 0.06167 | 0.01041 | 0.1817 | 0.03175 | 14.7 | 2.2 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | independent | 0.5237 | 0.007247 | 0.4974 | 0.002085 | 0.06667 | 0.02021 | 0.04 | 0.02784 | 23.51 | 3.434 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | joint | 0.5985 | 0.02022 | 0.5716 | 0.02186 | 0.06333 | 0.01756 | 0.09 | 0.02646 | 22.88 | 3.574 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | marginal_max | 0.5572 | 0.01513 | 0.5393 | 0.007436 | 0.05833 | 0.005774 | 0.07333 | 0.01258 | 14.33 | 2.716 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | relational_max | 0.7156 | 0.03638 | 0.7519 | 0.02962 | 0.04333 | 0.02021 | 0.3383 | 0.02517 | 5.453 | 1.167 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | conditional_max | 0.4892 | 0.004515 | 0.4987 | 0.006854 | 0.055 | 0.02 | 0.05833 | 0.01041 | 12.23 | 3.002 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | independent | 0.4892 | 0.004515 | 0.4987 | 0.006854 | 0.055 | 0.02 | 0.05833 | 0.01041 | 12.23 | 3.002 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | joint | 0.4892 | 0.004515 | 0.4987 | 0.006854 | 0.055 | 0.02 | 0.05833 | 0.01041 | 12.23 | 3.002 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | marginal_max | 0.4892 | 0.004515 | 0.4987 | 0.006854 | 0.055 | 0.02 | 0.05833 | 0.01041 | 12.23 | 3.002 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | conditional_max | 0.4892 | 0.004515 | 0.4987 | 0.006854 | 0.055 | 0.02 | 0.05833 | 0.01041 | 12.23 | 3.002 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | independent | 0.4892 | 0.004515 | 0.4987 | 0.006854 | 0.055 | 0.02 | 0.05833 | 0.01041 | 12.23 | 3.002 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | joint | 0.4892 | 0.004515 | 0.4987 | 0.006854 | 0.055 | 0.02 | 0.05833 | 0.01041 | 12.23 | 3.002 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | marginal_max | 0.4892 | 0.004515 | 0.4987 | 0.006854 | 0.055 | 0.02 | 0.05833 | 0.01041 | 12.23 | 3.002 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| blocked | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.055 | 0.03122 | 0.055 | 0.03122 | 13.92 | 4.624 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.05107 | 0.05667 | 0.05107 | 23.29 | 6.323 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.06167 | 0.05204 | 0.06167 | 0.05204 | 22.97 | 6.201 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.03775 | 0.06 | 0.03775 | 14.15 | 4.774 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.07167 | 0.02021 | 0.07167 | 0.02021 | 2.715 | 0.2757 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | conditional_max | 0.6512 | 0.009378 | 0.6522 | 0.01345 | 0.04167 | 0.02255 | 0.1267 | 0.05132 | 15.6 | 3.008 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | independent | 0.5244 | 0.002589 | 0.4971 | 0.001679 | 0.05 | 0.03905 | 0.03 | 0.02646 | 45.35 | 13.06 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | joint | 0.5991 | 0.008465 | 0.5589 | 0.0061 | 0.05167 | 0.04193 | 0.06667 | 0.05686 | 43.45 | 13.89 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | marginal_max | 0.566 | 0.007387 | 0.5433 | 0.001021 | 0.04 | 0.02646 | 0.04833 | 0.04163 | 16.28 | 2.432 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | relational_max | 0.718 | 0.02248 | 0.7467 | 0.01122 | 0.075 | 0.05679 | 0.3283 | 0.06252 | 7.363 | 0.7473 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.04833 | 0.03819 | 0.04833 | 0.03819 | 13.89 | 5.715 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.045 | 0.035 | 0.045 | 0.035 | 35.35 | 13.4 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.05575 | 0.05667 | 0.05575 | 34.1 | 13.84 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.04667 | 0.02754 | 0.04667 | 0.02754 | 14.44 | 5.305 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06667 | 0.02466 | 0.06667 | 0.02466 | 3.838 | 0.4589 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | conditional_max | 0.6472 | 0.02144 | 0.6479 | 0.02736 | 0.06167 | 0.002887 | 0.195 | 0.02291 | 13.97 | 2.185 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | independent | 0.5246 | 0.007148 | 0.4963 | 0.001993 | 0.07167 | 0.03055 | 0.04333 | 0.0293 | 23.19 | 3.835 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | joint | 0.595 | 0.01437 | 0.5669 | 0.01509 | 0.07 | 0.02 | 0.08833 | 0.02255 | 22.43 | 4.094 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | marginal_max | 0.5608 | 0.01602 | 0.5404 | 0.008248 | 0.065 | 0.01323 | 0.075 | 0.02784 | 13.88 | 2.25 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | relational_max | 0.7071 | 0.03321 | 0.7413 | 0.02108 | 0.04833 | 0.03055 | 0.2967 | 0.07286 | 5.515 | 1.268 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | conditional_max | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | independent | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | joint | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | marginal_max | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | conditional_max | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | independent | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | joint | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | marginal_max | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.055 | 0.03122 | 0.055 | 0.03122 | 13.92 | 4.624 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.05107 | 0.05667 | 0.05107 | 23.29 | 6.323 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.06167 | 0.05204 | 0.06167 | 0.05204 | 22.97 | 6.201 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.03775 | 0.06 | 0.03775 | 14.15 | 4.774 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.07167 | 0.02021 | 0.07167 | 0.02021 | 2.715 | 0.2757 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | conditional_max | 0.6512 | 0.009378 | 0.6522 | 0.01345 | 0.04167 | 0.02255 | 0.1267 | 0.05132 | 15.6 | 3.008 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | independent | 0.5244 | 0.002589 | 0.4971 | 0.001679 | 0.05 | 0.03905 | 0.03 | 0.02646 | 45.35 | 13.06 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | joint | 0.5991 | 0.008465 | 0.5589 | 0.0061 | 0.05167 | 0.04193 | 0.06667 | 0.05686 | 43.45 | 13.89 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | marginal_max | 0.566 | 0.007387 | 0.5433 | 0.001021 | 0.04 | 0.02646 | 0.04833 | 0.04163 | 16.28 | 2.432 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | relational_max | 0.718 | 0.02248 | 0.7467 | 0.01122 | 0.075 | 0.05679 | 0.3283 | 0.06252 | 7.363 | 0.7473 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.04833 | 0.03819 | 0.04833 | 0.03819 | 13.89 | 5.715 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.045 | 0.035 | 0.045 | 0.035 | 35.35 | 13.4 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.05575 | 0.05667 | 0.05575 | 34.1 | 13.84 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.04667 | 0.02754 | 0.04667 | 0.02754 | 14.44 | 5.305 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06667 | 0.02466 | 0.06667 | 0.02466 | 3.838 | 0.4589 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | conditional_max | 0.6472 | 0.02144 | 0.6479 | 0.02736 | 0.06167 | 0.002887 | 0.195 | 0.02291 | 13.97 | 2.185 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | independent | 0.5246 | 0.007148 | 0.4963 | 0.001993 | 0.07167 | 0.03055 | 0.04333 | 0.0293 | 23.19 | 3.835 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | joint | 0.595 | 0.01437 | 0.5669 | 0.01509 | 0.07 | 0.02 | 0.08833 | 0.02255 | 22.43 | 4.094 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | marginal_max | 0.5608 | 0.01602 | 0.5404 | 0.008248 | 0.065 | 0.01323 | 0.075 | 0.02784 | 13.88 | 2.25 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | relational_max | 0.7071 | 0.03321 | 0.7413 | 0.02108 | 0.04833 | 0.03055 | 0.2967 | 0.07286 | 5.515 | 1.268 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | conditional_max | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | independent | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | joint | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | marginal_max | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | conditional_max | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | independent | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | joint | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | marginal_max | 0.4898 | 0.002352 | 0.4987 | 0.003636 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 12.28 | 2.771 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.06167 | 0.03215 | 0.06167 | 0.03215 | 14.13 | 4.793 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.06333 | 0.05485 | 0.06333 | 0.05485 | 23.53 | 6.466 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.06167 | 0.04041 | 0.06167 | 0.04041 | 22.92 | 6.346 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.03606 | 0.06 | 0.03606 | 14.71 | 4.971 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06667 | 0.01258 | 0.06667 | 0.01258 | 3.328 | 0.4441 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | conditional_max | 0.6583 | 0.01333 | 0.6576 | 0.0214 | 0.04833 | 0.03253 | 0.1767 | 0.09878 | 15.83 | 2.444 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | independent | 0.5223 | 0.002877 | 0.4971 | 0.001945 | 0.05667 | 0.04368 | 0.03667 | 0.03403 | 45.28 | 11.74 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | joint | 0.5971 | 0.008623 | 0.5627 | 0.01493 | 0.05167 | 0.04537 | 0.07333 | 0.06934 | 43.38 | 12.37 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | marginal_max | 0.5667 | 0.01183 | 0.5454 | 0.004653 | 0.05167 | 0.03686 | 0.06667 | 0.04933 | 16.43 | 2.488 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | relational_max | 0.7178 | 0.01951 | 0.7469 | 0.03475 | 0.06833 | 0.03753 | 0.32 | 0.03969 | 8.389 | 0.7286 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.05167 | 0.04509 | 0.05167 | 0.04509 | 13.92 | 4.968 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.05333 | 0.04163 | 0.05333 | 0.04163 | 34.85 | 11.16 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.05 | 0.04272 | 0.05 | 0.04272 | 33.8 | 11.65 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.05167 | 0.04311 | 0.05167 | 0.04311 | 14.53 | 5.091 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.08333 | 0.0293 | 0.08333 | 0.0293 | 3.948 | 0.8727 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | conditional_max | 0.6427 | 0.0149 | 0.6388 | 0.009657 | 0.07333 | 0.01258 | 0.1883 | 0.04193 | 14.11 | 1.911 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | independent | 0.5208 | 0.008528 | 0.4972 | 0.001888 | 0.075 | 0.03041 | 0.04167 | 0.02517 | 23.86 | 4.306 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | joint | 0.587 | 0.009778 | 0.5599 | 0.007287 | 0.06333 | 0.02021 | 0.07667 | 0.03329 | 23.44 | 4.213 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | marginal_max | 0.56 | 0.0167 | 0.5421 | 0.008553 | 0.06333 | 0.01258 | 0.07333 | 0.01607 | 13.94 | 2.503 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | relational_max | 0.6936 | 0.007378 | 0.7307 | 0.01606 | 0.05667 | 0.01756 | 0.2883 | 0.01041 | 5.282 | 1.046 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | conditional_max | 0.4878 | 0.004735 | 0.5006 | 0.00465 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 12.9 | 3.047 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | independent | 0.4878 | 0.004735 | 0.5006 | 0.00465 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 12.9 | 3.047 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | joint | 0.4878 | 0.004735 | 0.5006 | 0.00465 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 12.9 | 3.047 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | marginal_max | 0.4878 | 0.004735 | 0.5006 | 0.00465 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 12.9 | 3.047 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | conditional_max | 0.4878 | 0.004735 | 0.5006 | 0.00465 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 12.9 | 3.047 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | independent | 0.4878 | 0.004735 | 0.5006 | 0.00465 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 12.9 | 3.047 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | joint | 0.4878 | 0.004735 | 0.5006 | 0.00465 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 12.9 | 3.047 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | marginal_max | 0.4878 | 0.004735 | 0.5006 | 0.00465 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 12.9 | 3.047 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.03786 | 0.05833 | 0.03786 | 13.56 | 4.799 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.05167 | 0.03014 | 0.05167 | 0.03014 | 23.13 | 6.25 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.0401 | 0.05833 | 0.0401 | 22.74 | 6.013 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.075 | 0.04272 | 0.075 | 0.04272 | 13.88 | 5.034 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.007638 | 0.05833 | 0.007638 | 3.23 | 0.05092 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | conditional_max | 0.6628 | 0.01408 | 0.6617 | 0.02004 | 0.05833 | 0.0401 | 0.1817 | 0.09074 | 15.63 | 1.569 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | independent | 0.5238 | 0.003301 | 0.4974 | 0.001365 | 0.05 | 0.02291 | 0.03333 | 0.01443 | 45.59 | 10.72 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | joint | 0.606 | 0.007202 | 0.5679 | 0.009917 | 0.065 | 0.05408 | 0.08 | 0.06874 | 43.63 | 10.87 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | marginal_max | 0.5689 | 0.01311 | 0.5444 | 0.005206 | 0.04667 | 0.03329 | 0.06167 | 0.04537 | 16.42 | 1.624 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | relational_max | 0.7235 | 0.01306 | 0.7497 | 0.0261 | 0.07833 | 0.0611 | 0.3467 | 0.01443 | 8.52 | 0.7713 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.04752 | 0.05667 | 0.04752 | 13.45 | 4.95 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.05167 | 0.03786 | 0.05167 | 0.03786 | 34.77 | 11.62 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.0562 | 0.05833 | 0.0562 | 33.56 | 11.88 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.04167 | 0.03329 | 0.04167 | 0.03329 | 13.84 | 5.1 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.065 | 0.05408 | 0.065 | 0.05408 | 3.886 | 0.3273 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | conditional_max | 0.6683 | 0.00916 | 0.6698 | 0.01682 | 0.075 | 0.015 | 0.2333 | 0.03819 | 15.06 | 1.473 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | independent | 0.5236 | 0.01027 | 0.4976 | 0.001533 | 0.055 | 0.005 | 0.03 | 0.01803 | 24.02 | 3.951 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | joint | 0.6089 | 0.004691 | 0.5829 | 0.005439 | 0.065 | 0.01323 | 0.07833 | 0.02466 | 23.32 | 3.918 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | marginal_max | 0.5604 | 0.01881 | 0.5416 | 0.0107 | 0.06833 | 0.005774 | 0.07833 | 0.01756 | 14.37 | 2.716 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | relational_max | 0.7265 | 0.02048 | 0.7604 | 0.02274 | 0.06 | 0.035 | 0.345 | 0.025 | 6.607 | 0.813 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | conditional_max | 0.4881 | 0.006176 | 0.501 | 0.006561 | 0.055 | 0.01323 | 0.055 | 0.005 | 13.05 | 3.912 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | independent | 0.4881 | 0.006176 | 0.501 | 0.006561 | 0.055 | 0.01323 | 0.055 | 0.005 | 13.05 | 3.912 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | joint | 0.4881 | 0.006176 | 0.501 | 0.006561 | 0.055 | 0.01323 | 0.055 | 0.005 | 13.05 | 3.912 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | marginal_max | 0.4881 | 0.006176 | 0.501 | 0.006561 | 0.055 | 0.01323 | 0.055 | 0.005 | 13.05 | 3.912 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | conditional_max | 0.4881 | 0.006176 | 0.501 | 0.006561 | 0.055 | 0.01323 | 0.055 | 0.005 | 13.05 | 3.912 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | independent | 0.4881 | 0.006176 | 0.501 | 0.006561 | 0.055 | 0.01323 | 0.055 | 0.005 | 13.05 | 3.912 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | joint | 0.4881 | 0.006176 | 0.501 | 0.006561 | 0.055 | 0.01323 | 0.055 | 0.005 | 13.05 | 3.912 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | marginal_max | 0.4881 | 0.006176 | 0.501 | 0.006561 | 0.055 | 0.01323 | 0.055 | 0.005 | 13.05 | 3.912 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface_conditional | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.03122 | 0.06 | 0.03122 | 14.2 | 4.791 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.02754 | 0.05833 | 0.02754 | 24.26 | 6.431 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.01443 | 0.05667 | 0.01443 | 23.25 | 6.684 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.07 | 0.05268 | 0.07 | 0.05268 | 15.16 | 4.651 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06167 | 0.01258 | 0.06167 | 0.01258 | 4.828 | 1.228 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | conditional_max | 0.7197 | 0.02245 | 0.718 | 0.01119 | 0.06 | 0.03606 | 0.25 | 0.1282 | 16.46 | 1.924 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | independent | 0.5253 | 0.005551 | 0.4974 | 0.000553 | 0.04333 | 0.01041 | 0.02833 | 0.01443 | 45.38 | 11.07 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | joint | 0.6579 | 0.01099 | 0.6133 | 0.00457 | 0.05667 | 0.05107 | 0.1033 | 0.1005 | 42.86 | 11.35 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | marginal_max | 0.57 | 0.01654 | 0.5466 | 0.006795 | 0.055 | 0.02 | 0.06667 | 0.03686 | 16.99 | 1.897 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | relational_max | 0.7886 | 0.02796 | 0.8067 | 0.02697 | 0.065 | 0.025 | 0.395 | 0.03775 | 10.25 | 0.6339 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.04193 | 0.05833 | 0.04193 | 13.88 | 5.015 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.04167 | 0.02021 | 0.04167 | 0.02021 | 36.08 | 11.65 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.04167 | 0.03253 | 0.04167 | 0.03253 | 33.65 | 12.5 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.055 | 0.03775 | 0.055 | 0.03775 | 15.06 | 4.731 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.01732 | 0.06 | 0.01732 | 6.571 | 0.9355 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | conditional_max | 0.7252 | 0.01757 | 0.7262 | 0.02973 | 0.08 | 0.01732 | 0.3267 | 0.03403 | 16.55 | 2.859 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | independent | 0.5244 | 0.01375 | 0.4978 | 0.001953 | 0.05333 | 0.005774 | 0.035 | 0.02 | 23.56 | 4.972 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | joint | 0.6594 | 0.01236 | 0.6339 | 0.02012 | 0.06667 | 0.01041 | 0.1383 | 0.06007 | 22.81 | 5.269 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | marginal_max | 0.5602 | 0.02067 | 0.5422 | 0.0114 | 0.06833 | 0.01443 | 0.09 | 0.025 | 14.67 | 2.515 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | relational_max | 0.8007 | 0.02423 | 0.8223 | 0.01657 | 0.075 | 0.02598 | 0.4633 | 0.04646 | 8.001 | 0.7414 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | conditional_max | 0.4853 | 0.008822 | 0.501 | 0.007688 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 12.56 | 3.785 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | independent | 0.4853 | 0.008822 | 0.501 | 0.007688 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 12.56 | 3.785 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | joint | 0.4853 | 0.008822 | 0.501 | 0.007688 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 12.56 | 3.785 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | marginal_max | 0.4853 | 0.008822 | 0.501 | 0.007688 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 12.56 | 3.785 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | conditional_max | 0.4853 | 0.008822 | 0.501 | 0.007688 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 12.56 | 3.785 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | independent | 0.4853 | 0.008822 | 0.501 | 0.007688 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 12.56 | 3.785 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | joint | 0.4853 | 0.008822 | 0.501 | 0.007688 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 12.56 | 3.785 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | marginal_max | 0.4853 | 0.008822 | 0.501 | 0.007688 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 12.56 | 3.785 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| mixture | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.055 | 0.02784 | 0.055 | 0.02784 | 15.03 | 6.455 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.06167 | 0.01258 | 0.06167 | 0.01258 | 25.56 | 6.39 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.01323 | 0.06 | 0.01323 | 24.71 | 6.782 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.05333 | 0.02021 | 0.05333 | 0.02021 | 15.69 | 5.883 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.05 | 0.005 | 0.05 | 0.005 | 4.74 | 0.7871 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | conditional_max | 0.7075 | 0.03076 | 0.7231 | 0.03848 | 0.04667 | 0.02082 | 0.2633 | 0.02466 | 21.39 | 4.588 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | independent | 0.527 | 0.002478 | 0.5019 | 0.0008657 | 0.055 | 0.02179 | 0.03333 | 0.01893 | 45.34 | 10.72 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | joint | 0.6579 | 0.03543 | 0.6291 | 0.03846 | 0.065 | 0.03905 | 0.1233 | 0.04752 | 47.24 | 4.539 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | marginal_max | 0.5581 | 0.01523 | 0.5393 | 0.007597 | 0.055 | 0.02179 | 0.06833 | 0.01893 | 17.5 | 2.892 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | relational_max | 0.7623 | 0.02852 | 0.7891 | 0.02989 | 0.07333 | 0.0293 | 0.4233 | 0.1068 | 17.19 | 4.383 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.055 | 0.02784 | 0.055 | 0.02784 | 15.31 | 6.871 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.01258 | 0.05667 | 0.01258 | 36.99 | 10.03 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.045 | 0.01732 | 0.045 | 0.01732 | 35.86 | 10.91 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.055 | 0.01323 | 0.055 | 0.01323 | 15.63 | 5.855 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.035 | 0.06 | 0.035 | 6.146 | 0.784 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | conditional_max | 0.6853 | 0.01956 | 0.686 | 0.04172 | 0.06667 | 0.007638 | 0.245 | 0.06946 | 19.97 | 1.578 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | independent | 0.5251 | 0.006978 | 0.5021 | 0.002717 | 0.055 | 0.03041 | 0.04 | 0.01323 | 24.26 | 5.918 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | joint | 0.6327 | 0.01633 | 0.6109 | 0.02401 | 0.05667 | 0.02082 | 0.1217 | 0.06526 | 25.49 | 4.753 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | marginal_max | 0.5568 | 0.01355 | 0.5378 | 0.01041 | 0.06667 | 0.01041 | 0.06833 | 0.01528 | 15.74 | 4.998 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | relational_max | 0.7353 | 0.03284 | 0.7701 | 0.02735 | 0.06667 | 0.01756 | 0.3683 | 0.08893 | 11.16 | 3.602 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | conditional_max | 0.4888 | 0.005235 | 0.5011 | 0.006514 | 0.065 | 0.015 | 0.055 | 0.01803 | 11.96 | 3.259 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | independent | 0.4888 | 0.005235 | 0.5011 | 0.006514 | 0.065 | 0.015 | 0.055 | 0.01803 | 11.96 | 3.259 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | joint | 0.4888 | 0.005235 | 0.5011 | 0.006514 | 0.065 | 0.015 | 0.055 | 0.01803 | 11.96 | 3.259 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | marginal_max | 0.4888 | 0.005235 | 0.5011 | 0.006514 | 0.065 | 0.015 | 0.055 | 0.01803 | 11.96 | 3.259 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | conditional_max | 0.4888 | 0.005235 | 0.5011 | 0.006514 | 0.065 | 0.015 | 0.055 | 0.01803 | 11.96 | 3.259 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | independent | 0.4888 | 0.005235 | 0.5011 | 0.006514 | 0.065 | 0.015 | 0.055 | 0.01803 | 11.96 | 3.259 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | joint | 0.4888 | 0.005235 | 0.5011 | 0.006514 | 0.065 | 0.015 | 0.055 | 0.01803 | 11.96 | 3.259 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | marginal_max | 0.4888 | 0.005235 | 0.5011 | 0.006514 | 0.065 | 0.015 | 0.055 | 0.01803 | 11.96 | 3.259 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| random | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.07833 | 0.01756 | 0.07833 | 0.01756 | 12.89 | 4.244 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.065 | 0.01323 | 0.065 | 0.01323 | 23.76 | 5.843 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.06333 | 0.01041 | 0.06333 | 0.01041 | 22.01 | 6.294 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.07 | 0.02784 | 0.07 | 0.02784 | 14.35 | 4.303 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06833 | 0.01607 | 0.06833 | 0.01607 | 9.267 | 2.325 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | conditional_max | 0.8205 | 0.01175 | 0.8479 | 0.006729 | 0.04833 | 0.005774 | 0.4883 | 0.01258 | 34.34 | 1.189 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | independent | 0.5272 | 0.003205 | 0.5021 | 0.001557 | 0.05667 | 0.007638 | 0.045 | 0.01803 | 44.95 | 10.64 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | joint | 0.7869 | 0.001175 | 0.792 | 0.009907 | 0.05667 | 0.01258 | 0.365 | 0.1212 | 47.9 | 5.197 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | marginal_max | 0.5683 | 0.006604 | 0.5472 | 0.005356 | 0.06667 | 0.03253 | 0.08167 | 0.03753 | 16.46 | 1.307 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | relational_max | 0.8594 | 0.01592 | 0.8761 | 0.01411 | 0.06333 | 0.01258 | 0.5417 | 0.06788 | 31.81 | 0.5205 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.05333 | 0.02517 | 0.05333 | 0.02517 | 12.95 | 3.788 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.01443 | 0.05833 | 0.01443 | 35.37 | 10.73 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.07667 | 0.01041 | 0.07667 | 0.01041 | 31.53 | 11.77 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.06667 | 0.02466 | 0.06667 | 0.02466 | 14.33 | 4.293 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.01 | 0.06 | 0.01 | 10.66 | 2.005 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | conditional_max | 0.8213 | 0.02239 | 0.8399 | 0.02188 | 0.06667 | 0.01528 | 0.5017 | 0.03175 | 27.65 | 1.907 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | independent | 0.5296 | 0.008792 | 0.5042 | 0.002688 | 0.06333 | 0.03215 | 0.04167 | 0.02566 | 23.72 | 3.295 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | joint | 0.764 | 0.01657 | 0.775 | 0.01942 | 0.055 | 0.005 | 0.34 | 0.03905 | 30.9 | 2.305 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | marginal_max | 0.5623 | 0.01017 | 0.5464 | 0.007704 | 0.065 | 0.00866 | 0.075 | 0.015 | 14.57 | 2.637 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | relational_max | 0.8738 | 0.02213 | 0.8836 | 0.02287 | 0.06333 | 0.03512 | 0.57 | 0.035 | 23.64 | 1.35 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | conditional_max | 0.4911 | 0.007025 | 0.505 | 0.0102 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 12.82 | 3.422 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | independent | 0.4911 | 0.007025 | 0.505 | 0.0102 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 12.82 | 3.422 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | joint | 0.4911 | 0.007025 | 0.505 | 0.0102 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 12.82 | 3.422 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | marginal_max | 0.4911 | 0.007025 | 0.505 | 0.0102 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 12.82 | 3.422 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | conditional_max | 0.4911 | 0.007025 | 0.505 | 0.0102 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 12.82 | 3.422 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | independent | 0.4911 | 0.007025 | 0.505 | 0.0102 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 12.82 | 3.422 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | joint | 0.4911 | 0.007025 | 0.505 | 0.0102 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 12.82 | 3.422 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | marginal_max | 0.4911 | 0.007025 | 0.505 | 0.0102 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 12.82 | 3.422 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| unblocked | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.065 | 0.02291 | 0.065 | 0.02291 | 13.84 | 4.522 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.07833 | 0.05008 | 0.07833 | 0.05008 | 23.37 | 5.578 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.08167 | 0.05204 | 0.08167 | 0.05204 | 22.88 | 5.432 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.08667 | 0.06658 | 0.08667 | 0.06658 | 14.22 | 4.546 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.02754 | 0.05667 | 0.02754 | 2.86 | 1.169 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | conditional_max | 0.656 | 0.01654 | 0.656 | 0.0206 | 0.05667 | 0.0401 | 0.16 | 0.07263 | 16.26 | 0.7896 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | independent | 0.523 | 0.002945 | 0.4956 | 0.001042 | 0.04833 | 0.03819 | 0.02667 | 0.02566 | 45.37 | 9.786 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | joint | 0.592 | 0.015 | 0.5537 | 0.01294 | 0.06667 | 0.05923 | 0.07333 | 0.08386 | 43.38 | 9.862 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | marginal_max | 0.5663 | 0.009776 | 0.544 | 0.003445 | 0.05167 | 0.03753 | 0.06333 | 0.04752 | 16.94 | 1.115 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | relational_max | 0.7073 | 0.03317 | 0.7347 | 0.02577 | 0.08167 | 0.04252 | 0.3317 | 0.06526 | 7.876 | 0.5551 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.05167 | 0.04041 | 0.05167 | 0.04041 | 13.62 | 4.443 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.055 | 0.05766 | 0.055 | 0.05766 | 34.83 | 11.22 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.06449 | 0.05833 | 0.06449 | 33.66 | 11.02 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.03014 | 0.05667 | 0.03014 | 14.16 | 4.643 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.065 | 0.03606 | 0.065 | 0.03606 | 4.203 | 0.06168 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | conditional_max | 0.6554 | 0.04265 | 0.6583 | 0.05376 | 0.07167 | 0.02021 | 0.2217 | 0.04481 | 14.87 | 3.174 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | independent | 0.5229 | 0.006227 | 0.4939 | 0.002068 | 0.07 | 0.03279 | 0.04167 | 0.02021 | 23.68 | 4.792 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | joint | 0.5999 | 0.02956 | 0.5743 | 0.03519 | 0.075 | 0.035 | 0.105 | 0.04093 | 22.66 | 4.82 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | marginal_max | 0.5581 | 0.0162 | 0.5401 | 0.008986 | 0.06833 | 0.03512 | 0.08 | 0.04272 | 14.6 | 3.897 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | relational_max | 0.7183 | 0.04888 | 0.7513 | 0.03996 | 0.06833 | 0.06007 | 0.3467 | 0.08145 | 5.865 | 1.829 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | conditional_max | 0.4876 | 0.004834 | 0.4981 | 0.007049 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 13.32 | 4.585 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | independent | 0.4876 | 0.004834 | 0.4981 | 0.007049 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 13.32 | 4.585 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | joint | 0.4876 | 0.004834 | 0.4981 | 0.007049 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 13.32 | 4.585 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | marginal_max | 0.4876 | 0.004834 | 0.4981 | 0.007049 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 13.32 | 4.585 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | conditional_max | 0.4876 | 0.004834 | 0.4981 | 0.007049 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 13.32 | 4.585 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | independent | 0.4876 | 0.004834 | 0.4981 | 0.007049 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 13.32 | 4.585 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | joint | 0.4876 | 0.004834 | 0.4981 | 0.007049 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 13.32 | 4.585 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | marginal_max | 0.4876 | 0.004834 | 0.4981 | 0.007049 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 13.32 | 4.585 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| uniform_wide | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | circuit | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.08833 | 0.05686 | 0.08833 | 0.05686 | 14.14 | 4.468 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | 0.07333 | 0.06028 | 0.07333 | 0.06028 | 24.03 | 6.231 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | 0.07667 | 0.06506 | 0.07667 | 0.06506 | 23.54 | 6.225 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.07333 | 0.05058 | 0.07333 | 0.05058 | 14.46 | 4.313 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.03167 | 0.03014 | 0.03167 | 0.03014 | 1.704 | 1.477 | 3 | 0.3333 | 0 | 0 | 0 |
| within_capacity | circuit | full | conditional_max | 0.6212 | 0.03187 | 0.6094 | 0.03973 | 0.04167 | 0.03403 | 0.08667 | 0.03547 | 15.7 | 1.184 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | independent | 0.5218 | 0.002578 | 0.4947 | 0.00208 | 0.07167 | 0.07286 | 0.04667 | 0.05107 | 45.72 | 10.16 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | joint | 0.5595 | 0.02843 | 0.5258 | 0.02065 | 0.06167 | 0.05923 | 0.05333 | 0.04311 | 44.76 | 11.42 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | marginal_max | 0.5678 | 0.008281 | 0.5454 | 0.002872 | 0.05 | 0.04093 | 0.06667 | 0.05838 | 16.42 | 1.15 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | relational_max | 0.6588 | 0.05531 | 0.6827 | 0.06285 | 0.05667 | 0.01607 | 0.2367 | 0.08401 | 5.224 | 2.449 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | 0.05833 | 0.05008 | 0.05833 | 0.05008 | 13.66 | 4.19 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | 0.05 | 0.05679 | 0.05 | 0.05679 | 35.3 | 11.09 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | 0.05167 | 0.04368 | 0.05167 | 0.04368 | 34.48 | 11.51 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | 0.06 | 0.05568 | 0.06 | 0.05568 | 14.39 | 4.281 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0.05667 | 0.02021 | 0.05667 | 0.02021 | 2.586 | 1.011 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | conditional_max | 0.5892 | 0.0594 | 0.5763 | 0.06145 | 0.05833 | 0.01528 | 0.095 | 0.01 | 14.29 | 2.53 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | independent | 0.5222 | 0.008017 | 0.495 | 0.002735 | 0.07167 | 0.04726 | 0.05 | 0.03122 | 24.03 | 4.135 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | joint | 0.5442 | 0.04564 | 0.5161 | 0.03764 | 0.07 | 0.04924 | 0.05833 | 0.01756 | 23.74 | 4.238 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | marginal_max | 0.5611 | 0.01195 | 0.5433 | 0.007225 | 0.05667 | 0.01756 | 0.08 | 0.02784 | 14.52 | 2.733 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | relational_max | 0.5401 | 0.1342 | 0.561 | 0.1343 | 0.05 | 0.035 | 0.125 | 0.1612 | 1.879 | 3.254 | 3 | 0.6667 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | conditional_max | 0.4858 | 0.006741 | 0.4993 | 0.006062 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 13.11 | 3.76 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | independent | 0.4858 | 0.006741 | 0.4993 | 0.006062 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 13.11 | 3.76 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | joint | 0.4858 | 0.006741 | 0.4993 | 0.006062 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 13.11 | 3.76 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | marginal_max | 0.4858 | 0.006741 | 0.4993 | 0.006062 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 13.11 | 3.76 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | conditional_max | 0.4858 | 0.006741 | 0.4993 | 0.006062 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 13.11 | 3.76 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | independent | 0.4858 | 0.006741 | 0.4993 | 0.006062 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 13.11 | 3.76 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | joint | 0.4858 | 0.006741 | 0.4993 | 0.006062 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 13.11 | 3.76 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | marginal_max | 0.4858 | 0.006741 | 0.4993 | 0.006062 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 13.11 | 3.76 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| within_capacity | known_law_oracle | bank0:0-1 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | conditional_max | 0.9254 | 0.007314 | 0.9409 | 0.0063 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | independent | 0.5234 | 0.004127 | 0.4953 | 0.004302 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | joint | 0.8885 | 0.002434 | 0.9133 | 0.001032 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | marginal_max | 0.5723 | 0.00311 | 0.5453 | 0.009046 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | relational_max | 0.9431 | 0.01282 | 0.9552 | 0.0095 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | conditional_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | independent | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | joint | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | marginal_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | conditional_max | 0.8997 | 0.01745 | 0.9216 | 0.01225 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | independent | 0.5248 | 0.002194 | 0.4943 | 0.008234 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | joint | 0.848 | 0.01524 | 0.8797 | 0.008831 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | marginal_max | 0.5665 | 0.01171 | 0.5462 | 0.004068 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | relational_max | 0.9314 | 0.01462 | 0.9447 | 0.0105 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | conditional_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | independent | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | joint | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | marginal_max | 0.4866 | 0.006085 | 0.4984 | 0.008169 | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | relational_max | 0.5 | 0 | 0.5 | 0 | — | — | — | — | — | — | 3 | — | — | — | — |

## Localisation

| variant | method_name | mask | score | loc_ap_n_mean | loc_ap_n_std | end_to_end_unique_top1_mean | end_to_end_unique_top1_std | loc_n_ambiguous_mean | loc_n_ambiguous_std | loc_n_abstained_no_alarm_mean | loc_n_abstained_no_alarm_std | loc_n_no_observed_target_mean | loc_n_no_observed_target_std | loc_n_all_observed_are_targets_mean | loc_n_all_observed_are_targets_std | n_seeds | numerically_flat_frac | threshold_infinite_frac | depends_on_values_frac | depends_on_fault_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocked | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0.3333 | 0 | 0 | 0 |
| blocked | circuit | full | conditional_max | 200 | 0 | 0.09667 | 0.04646 | 0 | 0 | 170 | 13.11 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | independent | 200 | 0 | 0.005 | 0.005 | 0 | 0 | 193.3 | 4.619 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | joint | 200 | 0 | 0.03167 | 0.02566 | 0 | 0 | 187.3 | 11.24 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | marginal_max | 200 | 0 | 0.02 | 0.015 | 0 | 0 | 188.7 | 9.452 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | relational_max | 200 | 0 | 0.1617 | 0.14 | 65.67 | 113.7 | 93.33 | 80.85 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0.3333 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.085 | 0.02179 | 0 | 0 | 163.7 | 6.351 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | independent | 200 | 0 | 0.02 | 0.01323 | 0 | 0 | 192 | 5.568 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | joint | 200 | 0 | 0.05 | 0.00866 | 0 | 0 | 182 | 5.292 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.03667 | 0.002887 | 0 | 0 | 185.3 | 2.517 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| blocked | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | conditional_max | 200 | 0 | 0.09333 | 0.02466 | 0 | 0 | 174.7 | 10.26 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | independent | 200 | 0 | 0.005 | 0.005 | 0 | 0 | 194 | 5.292 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | joint | 200 | 0 | 0.03 | 0.01803 | 0 | 0 | 186.7 | 11.37 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | marginal_max | 200 | 0 | 0.01833 | 0.01528 | 0 | 0 | 190.3 | 8.327 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | relational_max | 200 | 0 | 0.265 | 0.0433 | 0 | 0 | 134.3 | 12.5 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.09667 | 0.01756 | 0 | 0 | 161 | 4.583 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | independent | 200 | 0 | 0.02833 | 0.02082 | 0 | 0 | 191.3 | 5.859 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | joint | 200 | 0 | 0.04667 | 0.01893 | 0 | 0 | 182.3 | 4.509 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.04 | 0.01 | 0 | 0 | 185 | 5.568 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | conditional_max | 200 | 0 | 0.09333 | 0.02466 | 0 | 0 | 174.7 | 10.26 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | independent | 200 | 0 | 0.005 | 0.005 | 0 | 0 | 194 | 5.292 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | joint | 200 | 0 | 0.03 | 0.01803 | 0 | 0 | 186.7 | 11.37 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | marginal_max | 200 | 0 | 0.01833 | 0.01528 | 0 | 0 | 190.3 | 8.327 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | relational_max | 200 | 0 | 0.265 | 0.0433 | 0 | 0 | 134.3 | 12.5 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.09667 | 0.01756 | 0 | 0 | 161 | 4.583 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | independent | 200 | 0 | 0.02833 | 0.02082 | 0 | 0 | 191.3 | 5.859 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | joint | 200 | 0 | 0.04667 | 0.01893 | 0 | 0 | 182.3 | 4.509 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.04 | 0.01 | 0 | 0 | 185 | 5.568 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | conditional_max | 200 | 0 | 0.1283 | 0.05575 | 0 | 0 | 164.7 | 19.76 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | independent | 200 | 0 | 0.003333 | 0.005774 | 0 | 0 | 192.7 | 6.807 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | joint | 200 | 0 | 0.04333 | 0.04646 | 0 | 0 | 185.3 | 13.87 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | marginal_max | 200 | 0 | 0.02333 | 0.01607 | 0 | 0 | 186.7 | 9.866 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | relational_max | 200 | 0 | 0.2483 | 0.03753 | 0 | 0 | 136 | 7.937 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.09167 | 0.007638 | 0 | 0 | 162.3 | 8.386 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | independent | 200 | 0 | 0.02 | 0.01 | 0 | 0 | 191.7 | 5.033 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | joint | 200 | 0 | 0.04167 | 0.01528 | 0 | 0 | 184.7 | 6.658 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.035 | 0.005 | 0 | 0 | 185.3 | 3.215 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | conditional_max | 200 | 0 | 0.13 | 0.05292 | 0 | 0 | 163.7 | 18.15 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | independent | 200 | 0 | 0.005 | 0.005 | 0 | 0 | 193.3 | 2.887 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | joint | 200 | 0 | 0.045 | 0.03606 | 0 | 0 | 184 | 13.75 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | marginal_max | 200 | 0 | 0.02167 | 0.01443 | 0 | 0 | 187.7 | 9.074 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | relational_max | 200 | 0 | 0.2667 | 0.02754 | 0 | 0 | 130.7 | 2.887 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.115 | 0.00866 | 0 | 0 | 153.3 | 7.638 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | independent | 200 | 0 | 0.01167 | 0.007638 | 0 | 0 | 194 | 3.606 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | joint | 200 | 0 | 0.04333 | 0.007638 | 0 | 0 | 184.3 | 4.933 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.035 | 0.005 | 0 | 0 | 184.3 | 3.512 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface_conditional | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | conditional_max | 200 | 0 | 0.1967 | 0.0878 | 0 | 0 | 150 | 25.63 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | independent | 200 | 0 | 0.003333 | 0.002887 | 0 | 0 | 194.3 | 2.887 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | joint | 200 | 0 | 0.065 | 0.05568 | 0 | 0 | 179.3 | 20.11 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | marginal_max | 200 | 0 | 0.02167 | 0.01443 | 0 | 0 | 186.7 | 7.371 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | relational_max | 200 | 0 | 0.34 | 0.03775 | 0.3333 | 0.5774 | 120.7 | 8.083 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.1567 | 0.02255 | 0 | 0 | 134.7 | 6.807 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | independent | 200 | 0 | 0.01833 | 0.01041 | 0 | 0 | 193 | 4 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | joint | 200 | 0 | 0.06833 | 0.02517 | 0 | 0 | 172.3 | 12.01 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.04167 | 0.005774 | 0 | 0 | 182 | 5 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| mixture | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | conditional_max | 200 | 0 | 0.195 | 0.02 | 0 | 0 | 147.3 | 4.933 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | independent | 200 | 0 | 0.001667 | 0.002887 | 0 | 0 | 193.3 | 3.786 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | joint | 200 | 0 | 0.07167 | 0.02363 | 0 | 0 | 175.3 | 9.504 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | marginal_max | 200 | 0 | 0.02167 | 0.002887 | 0 | 0 | 186.3 | 3.786 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | relational_max | 200 | 0 | 0.3467 | 0.07112 | 0 | 0 | 115.3 | 21.36 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.1183 | 0.01528 | 0 | 0 | 151 | 13.89 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | independent | 200 | 0 | 0.02167 | 0.002887 | 0 | 0 | 192 | 2.646 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | joint | 200 | 0 | 0.065 | 0.02291 | 0 | 0 | 175.7 | 13.05 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.03 | 0.00866 | 0 | 0 | 186.3 | 3.055 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| random | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | conditional_max | 200 | 0 | 0.445 | 0.02598 | 0 | 0 | 102.3 | 2.517 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | independent | 200 | 0 | 0.008333 | 0.005774 | 0.3333 | 0.5774 | 190.7 | 3.512 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | joint | 200 | 0 | 0.3183 | 0.106 | 0 | 0 | 127 | 24.25 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | marginal_max | 200 | 0 | 0.03 | 0.00866 | 0.3333 | 0.5774 | 183.3 | 7.506 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | relational_max | 200 | 0 | 0.4967 | 0.06714 | 0 | 0 | 91.67 | 13.58 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.2367 | 0.01155 | 0.3333 | 0.5774 | 99.33 | 6.658 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | independent | 200 | 0 | 0.02167 | 0.01258 | 0.3333 | 0.5774 | 191.3 | 4.726 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | joint | 200 | 0 | 0.1633 | 0.007638 | 0.3333 | 0.5774 | 131.7 | 7.572 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.04167 | 0.002887 | 0.3333 | 0.5774 | 184.7 | 2.517 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| unblocked | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | conditional_max | 200 | 0 | 0.11 | 0.04583 | 0.3333 | 0.5774 | 167.7 | 14.05 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | independent | 200 | 0 | 0.003333 | 0.005774 | 0 | 0 | 194.7 | 5.132 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | joint | 200 | 0 | 0.03333 | 0.03175 | 0.3333 | 0.5774 | 185 | 16.46 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | marginal_max | 200 | 0 | 0.02167 | 0.01528 | 0 | 0 | 187.3 | 9.504 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | relational_max | 200 | 0 | 0.2667 | 0.04537 | 0.3333 | 0.5774 | 133.3 | 13.61 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.1017 | 0.02363 | 0 | 0 | 155.7 | 8.963 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | independent | 200 | 0 | 0.02167 | 0.01041 | 0 | 0 | 191.7 | 4.041 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | joint | 200 | 0 | 0.05167 | 0.01258 | 0 | 0 | 179 | 8.185 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.03333 | 0.01258 | 0 | 0 | 184 | 8.544 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| uniform_wide | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | circuit | bank0:0-1 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0.3333 | 0 | 0 | 0 |
| within_capacity | circuit | full | conditional_max | 200 | 0 | 0.04833 | 0.005774 | 0 | 0 | 182.7 | 7.095 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | independent | 200 | 0 | 0.005 | 0.00866 | 0 | 0 | 190.7 | 10.21 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | joint | 200 | 0 | 0.015 | 0.01 | 0 | 0 | 189.3 | 8.622 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | marginal_max | 200 | 0 | 0.02167 | 0.01756 | 0 | 0 | 186.7 | 11.68 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | relational_max | 200 | 0 | 0.095 | 0.1238 | 102.7 | 100.1 | 68.33 | 69.01 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | conditional_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | independent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | joint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | marginal_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | relational_max | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | conditional_max | 200 | 0 | 0.04333 | 0.005774 | 0 | 0 | 181 | 2 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | independent | 200 | 0 | 0.025 | 0.015 | 0 | 0 | 190 | 6.245 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | joint | 200 | 0 | 0.03 | 0.00866 | 0 | 0 | 188.3 | 3.512 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | marginal_max | 200 | 0 | 0.03833 | 0.01258 | 0 | 0 | 184 | 5.568 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | relational_max | 200 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0.6667 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | conditional_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | independent | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | joint | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | marginal_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | relational_max | 0 | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 3 | 1 | 0 | 0 | 0 |
| within_capacity | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | — | — | — | — | 3 | — | — | — | — |

## Operational alarms

| variant | method_name | mask | score | alpha_mean | alpha_std | fpr_mean | fpr_std | power_mean | power_std | threshold_mean | threshold_std | n_seeds | numerically_flat_frac | threshold_infinite_frac | depends_on_values_frac | depends_on_fault_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocked | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.06167 | 0.03753 | 0.06167 | 0.03753 | 8.83 | 1.154 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.05833 | 0.0401 | 0.05833 | 0.0401 | 16.23 | 1.737 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.06167 | 0.04252 | 0.06167 | 0.04252 | 15.84 | 2.034 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.05 | 0.02291 | 0.05 | 0.02291 | 9.235 | 0.8293 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.03667 | 0.01443 | 0.03667 | 0.01443 | 0.4155 | 0.3601 | 3 | 0.3333 | 0 | 0 | 0 |
| blocked | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.04444 | 0.15 | 0.06557 | 9.645 | 1.16 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | independent | 0.05 | 8.498e-18 | 0.05333 | 0.02309 | 0.03333 | 0.02309 | 31.59 | 1.884 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | joint | 0.05 | 8.498e-18 | 0.05333 | 0.04646 | 0.06333 | 0.0562 | 31.01 | 3.608 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.04833 | 0.03617 | 0.05667 | 0.04726 | 9.971 | 1.044 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.05667 | 0.01443 | 0.3217 | 0.03884 | 1.393 | 0.3506 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.05667 | 0.04856 | 0.05667 | 0.04856 | 9.297 | 1.453 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.04833 | 0.03617 | 0.04833 | 0.03617 | 23.91 | 1.875 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.05333 | 0.04072 | 0.05333 | 0.04072 | 23.28 | 2.536 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.045 | 0.03606 | 0.045 | 0.03606 | 9.71 | 1.145 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.06333 | 0.01893 | 0.06333 | 0.01893 | 0.7734 | 0.6733 | 3 | 0.3333 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.06167 | 0.01041 | 0.1817 | 0.03175 | 8.517 | 0.4368 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.06667 | 0.02021 | 0.04 | 0.02784 | 15.28 | 1.063 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.06333 | 0.01756 | 0.09 | 0.02646 | 15.04 | 0.7279 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.05833 | 0.005774 | 0.07333 | 0.01258 | 8.854 | 0.3522 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.04333 | 0.02021 | 0.3383 | 0.02517 | 0.7172 | 0.2796 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.05833 | 0.01041 | 8.111 | 0.6078 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.05833 | 0.01041 | 8.111 | 0.6078 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.05833 | 0.01041 | 8.111 | 0.6078 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.05833 | 0.01041 | 8.111 | 0.6078 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.05833 | 0.01041 | 8.111 | 0.6078 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.05833 | 0.01041 | 8.111 | 0.6078 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.05833 | 0.01041 | 8.111 | 0.6078 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.05833 | 0.01041 | 8.111 | 0.6078 | 3 | 0 | 0 | 0 | 0 |
| blocked | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| blocked | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| blocked | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.03122 | 0.055 | 0.03122 | 8.746 | 0.9545 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.05667 | 0.05107 | 0.05667 | 0.05107 | 16.14 | 1.693 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.06167 | 0.05204 | 0.06167 | 0.05204 | 15.62 | 1.687 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.06 | 0.03775 | 0.06 | 0.03775 | 9.065 | 0.9702 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.07167 | 0.02021 | 0.07167 | 0.02021 | 0.6191 | 0.2691 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.04167 | 0.02255 | 0.1267 | 0.05132 | 9.555 | 0.7698 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | independent | 0.05 | 8.498e-18 | 0.05 | 0.03905 | 0.03 | 0.02646 | 31.92 | 2.813 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | joint | 0.05 | 8.498e-18 | 0.05167 | 0.04193 | 0.06667 | 0.05686 | 29.72 | 2.465 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.04 | 0.02646 | 0.04833 | 0.04163 | 10 | 1.057 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.075 | 0.05679 | 0.3283 | 0.06252 | 1.804 | 0.166 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.04833 | 0.03819 | 0.04833 | 0.03819 | 9.071 | 0.9972 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.045 | 0.035 | 0.045 | 0.035 | 23.97 | 2.085 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.05667 | 0.05575 | 0.05667 | 0.05575 | 22.41 | 2.172 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.04667 | 0.02754 | 0.04667 | 0.02754 | 9.416 | 0.8784 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.06667 | 0.02466 | 0.06667 | 0.02466 | 1.348 | 0.04339 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.06167 | 0.002887 | 0.195 | 0.02291 | 8.299 | 0.2307 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.07167 | 0.03055 | 0.04333 | 0.0293 | 15.49 | 1.121 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.07 | 0.02 | 0.08833 | 0.02255 | 14.96 | 0.7849 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.065 | 0.01323 | 0.075 | 0.02784 | 8.703 | 0.5569 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.04833 | 0.03055 | 0.2967 | 0.07286 | 0.7683 | 0.1778 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.03122 | 0.055 | 0.03122 | 8.746 | 0.9545 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.05667 | 0.05107 | 0.05667 | 0.05107 | 16.14 | 1.693 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.06167 | 0.05204 | 0.06167 | 0.05204 | 15.62 | 1.687 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.06 | 0.03775 | 0.06 | 0.03775 | 9.065 | 0.9702 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.07167 | 0.02021 | 0.07167 | 0.02021 | 0.6191 | 0.2691 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.04167 | 0.02255 | 0.1267 | 0.05132 | 9.555 | 0.7698 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | independent | 0.05 | 8.498e-18 | 0.05 | 0.03905 | 0.03 | 0.02646 | 31.92 | 2.813 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | joint | 0.05 | 8.498e-18 | 0.05167 | 0.04193 | 0.06667 | 0.05686 | 29.72 | 2.465 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.04 | 0.02646 | 0.04833 | 0.04163 | 10 | 1.057 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.075 | 0.05679 | 0.3283 | 0.06252 | 1.804 | 0.166 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.04833 | 0.03819 | 0.04833 | 0.03819 | 9.071 | 0.9972 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.045 | 0.035 | 0.045 | 0.035 | 23.97 | 2.085 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.05667 | 0.05575 | 0.05667 | 0.05575 | 22.41 | 2.172 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.04667 | 0.02754 | 0.04667 | 0.02754 | 9.416 | 0.8784 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.06667 | 0.02466 | 0.06667 | 0.02466 | 1.348 | 0.04339 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.06167 | 0.002887 | 0.195 | 0.02291 | 8.299 | 0.2307 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.07167 | 0.03055 | 0.04333 | 0.0293 | 15.49 | 1.121 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.07 | 0.02 | 0.08833 | 0.02255 | 14.96 | 0.7849 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.065 | 0.01323 | 0.075 | 0.02784 | 8.703 | 0.5569 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.04833 | 0.03055 | 0.2967 | 0.07286 | 0.7683 | 0.1778 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.07167 | 0.01756 | 0.06833 | 0.01041 | 7.902 | 0.6537 | 3 | 0 | 0 | 0 | 0 |
| conditional_nll_selection | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| conditional_nll_selection | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.06167 | 0.03215 | 0.06167 | 0.03215 | 8.371 | 0.6743 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.06333 | 0.05485 | 0.06333 | 0.05485 | 15.88 | 1.849 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.06167 | 0.04041 | 0.06167 | 0.04041 | 15.31 | 1.489 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.06 | 0.03606 | 0.06 | 0.03606 | 8.872 | 0.8092 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.06667 | 0.01258 | 0.06667 | 0.01258 | 0.6954 | 0.197 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.04833 | 0.03253 | 0.1767 | 0.09878 | 9.239 | 1.298 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | independent | 0.05 | 8.498e-18 | 0.05667 | 0.04368 | 0.03667 | 0.03403 | 31.53 | 3.251 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | joint | 0.05 | 8.498e-18 | 0.05167 | 0.04537 | 0.07333 | 0.06934 | 30.23 | 3.631 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.05167 | 0.03686 | 0.06667 | 0.04933 | 9.716 | 1.325 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.06833 | 0.03753 | 0.32 | 0.03969 | 1.631 | 0.1723 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.05167 | 0.04509 | 0.05167 | 0.04509 | 8.921 | 1.247 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.05333 | 0.04163 | 0.05333 | 0.04163 | 23.74 | 2.442 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.05 | 0.04272 | 0.05 | 0.04272 | 22.77 | 2.809 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.05167 | 0.04311 | 0.05167 | 0.04311 | 9.608 | 1.451 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.08333 | 0.0293 | 0.08333 | 0.0293 | 1.142 | 0.1414 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.07333 | 0.01258 | 0.1883 | 0.04193 | 8.079 | 0.5889 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.075 | 0.03041 | 0.04167 | 0.02517 | 15.42 | 1.075 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.06333 | 0.02021 | 0.07667 | 0.03329 | 14.77 | 0.8583 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.06333 | 0.01258 | 0.07333 | 0.01607 | 8.74 | 0.3434 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.05667 | 0.01756 | 0.2883 | 0.01041 | 0.6186 | 0.1835 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 7.881 | 0.755 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 7.881 | 0.755 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 7.881 | 0.755 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 7.881 | 0.755 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 7.881 | 0.755 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 7.881 | 0.755 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 7.881 | 0.755 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.06833 | 0.03617 | 0.07 | 0.02179 | 7.881 | 0.755 | 3 | 0 | 0 | 0 | 0 |
| interface | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.03786 | 0.05833 | 0.03786 | 8.485 | 0.9513 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.05167 | 0.03014 | 0.05167 | 0.03014 | 16.29 | 1.586 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.05833 | 0.0401 | 0.05833 | 0.0401 | 15.57 | 1.746 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.075 | 0.04272 | 0.075 | 0.04272 | 8.675 | 1.04 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.05833 | 0.007638 | 0.05833 | 0.007638 | 0.7413 | 0.04867 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.0401 | 0.1817 | 0.09074 | 9.288 | 1.165 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | independent | 0.05 | 8.498e-18 | 0.05 | 0.02291 | 0.03333 | 0.01443 | 31.43 | 2.069 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | joint | 0.05 | 8.498e-18 | 0.065 | 0.05408 | 0.08 | 0.06874 | 29.52 | 3.316 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.04667 | 0.03329 | 0.06167 | 0.04537 | 9.948 | 1.19 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.07833 | 0.0611 | 0.3467 | 0.01443 | 1.736 | 0.249 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.05667 | 0.04752 | 0.05667 | 0.04752 | 9.013 | 1.258 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.05167 | 0.03786 | 0.05167 | 0.03786 | 23.63 | 2.221 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.05833 | 0.0562 | 0.05833 | 0.0562 | 22.22 | 2.686 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.04167 | 0.03329 | 0.04167 | 0.03329 | 9.782 | 1.37 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.065 | 0.05408 | 0.065 | 0.05408 | 1.462 | 0.1995 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.075 | 0.015 | 0.2333 | 0.03819 | 7.957 | 0.4804 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.055 | 0.005 | 0.03 | 0.01803 | 15.69 | 0.3725 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.065 | 0.01323 | 0.07833 | 0.02466 | 14.94 | 0.6692 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.06833 | 0.005774 | 0.07833 | 0.01756 | 8.636 | 0.3017 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.06 | 0.035 | 0.345 | 0.025 | 0.7184 | 0.01837 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.005 | 8.008 | 0.5056 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.005 | 8.008 | 0.5056 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.005 | 8.008 | 0.5056 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.005 | 8.008 | 0.5056 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.005 | 8.008 | 0.5056 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.005 | 8.008 | 0.5056 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.005 | 8.008 | 0.5056 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.005 | 8.008 | 0.5056 | 3 | 0 | 0 | 0 | 0 |
| interface_conditional | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| interface_conditional | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| interface_conditional | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.06 | 0.03122 | 0.06 | 0.03122 | 7.875 | 0.5808 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.05833 | 0.02754 | 16 | 1.147 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.05667 | 0.01443 | 0.05667 | 0.01443 | 15.03 | 1.088 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.07 | 0.05268 | 0.07 | 0.05268 | 8.896 | 1.1 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.06167 | 0.01258 | 0.06167 | 0.01258 | 0.8069 | 0.2787 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.06 | 0.03606 | 0.25 | 0.1282 | 8.95 | 1.042 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | independent | 0.05 | 8.498e-18 | 0.04333 | 0.01041 | 0.02833 | 0.01443 | 31.62 | 1.539 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | joint | 0.05 | 8.498e-18 | 0.05667 | 0.05107 | 0.1033 | 0.1005 | 28.74 | 3.789 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.02 | 0.06667 | 0.03686 | 9.77 | 1.007 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.065 | 0.025 | 0.395 | 0.03775 | 2.102 | 0.2171 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.04193 | 0.05833 | 0.04193 | 8.668 | 1.161 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.04167 | 0.02021 | 0.04167 | 0.02021 | 23.99 | 1.456 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.04167 | 0.03253 | 0.04167 | 0.03253 | 22.15 | 2.492 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.03775 | 0.055 | 0.03775 | 9.537 | 1.322 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.06 | 0.01732 | 0.06 | 0.01732 | 1.843 | 0.2262 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.08 | 0.01732 | 0.3267 | 0.03403 | 7.758 | 0.4018 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.05333 | 0.005774 | 0.035 | 0.02 | 15.69 | 0.5618 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.06667 | 0.01041 | 0.1383 | 0.06007 | 14.58 | 0.667 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.06833 | 0.01443 | 0.09 | 0.025 | 8.721 | 0.4717 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.075 | 0.02598 | 0.4633 | 0.04646 | 0.7116 | 0.2641 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 8.193 | 0.5519 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 8.193 | 0.5519 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 8.193 | 0.5519 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 8.193 | 0.5519 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 8.193 | 0.5519 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 8.193 | 0.5519 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 8.193 | 0.5519 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.05833 | 0.02466 | 0.05667 | 0.01258 | 8.193 | 0.5519 | 3 | 0 | 0 | 0 | 0 |
| mixture | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| mixture | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| mixture | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.02784 | 0.055 | 0.02784 | 8.946 | 0.6334 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.06167 | 0.01258 | 0.06167 | 0.01258 | 16.11 | 0.1392 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.06 | 0.01323 | 0.06 | 0.01323 | 15.89 | 0.7193 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.05333 | 0.02021 | 0.05333 | 0.02021 | 9.483 | 0.2355 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.05 | 0.005 | 0.05 | 0.005 | 0.9769 | 0.1309 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.04667 | 0.02082 | 0.2633 | 0.02466 | 9.776 | 0.6608 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | independent | 0.05 | 8.498e-18 | 0.055 | 0.02179 | 0.03333 | 0.01893 | 32.23 | 1.463 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | joint | 0.05 | 8.498e-18 | 0.065 | 0.03905 | 0.1233 | 0.04752 | 29.57 | 1.926 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.02179 | 0.06833 | 0.01893 | 10.15 | 0.6092 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.07333 | 0.0293 | 0.4233 | 0.1068 | 2.127 | 0.08947 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.02784 | 0.055 | 0.02784 | 9.382 | 0.7751 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.05667 | 0.01258 | 0.05667 | 0.01258 | 24.18 | 0.4991 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.045 | 0.01732 | 0.045 | 0.01732 | 23.25 | 1.574 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.01323 | 0.055 | 0.01323 | 9.913 | 0.5487 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.06 | 0.035 | 0.06 | 0.035 | 1.895 | 0.04764 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.06667 | 0.007638 | 0.245 | 0.06946 | 8.599 | 0.1919 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.055 | 0.03041 | 0.04 | 0.01323 | 16.2 | 1.05 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.05667 | 0.02082 | 0.1217 | 0.06526 | 15.47 | 0.7057 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.06667 | 0.01041 | 0.06833 | 0.01528 | 9.174 | 0.3326 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.06667 | 0.01756 | 0.3683 | 0.08893 | 0.7111 | 0.06379 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.065 | 0.015 | 0.055 | 0.01803 | 8.326 | 0.6094 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.065 | 0.015 | 0.055 | 0.01803 | 8.326 | 0.6094 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.065 | 0.015 | 0.055 | 0.01803 | 8.326 | 0.6094 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.065 | 0.015 | 0.055 | 0.01803 | 8.326 | 0.6094 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.065 | 0.015 | 0.055 | 0.01803 | 8.326 | 0.6094 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.065 | 0.015 | 0.055 | 0.01803 | 8.326 | 0.6094 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.065 | 0.015 | 0.055 | 0.01803 | 8.326 | 0.6094 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.065 | 0.015 | 0.055 | 0.01803 | 8.326 | 0.6094 | 3 | 0 | 0 | 0 | 0 |
| random | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| random | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| random | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.07833 | 0.01756 | 0.07833 | 0.01756 | 7.565 | 0.418 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.065 | 0.01323 | 0.065 | 0.01323 | 15.41 | 0.6116 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.06333 | 0.01041 | 0.06333 | 0.01041 | 13.56 | 0.3898 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.07 | 0.02784 | 0.07 | 0.02784 | 8.83 | 0.4305 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.06833 | 0.01607 | 0.06833 | 0.01607 | 1.318 | 0.4647 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.04833 | 0.005774 | 0.4883 | 0.01258 | 9.155 | 0.2924 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | independent | 0.05 | 8.498e-18 | 0.05667 | 0.007638 | 0.045 | 0.01803 | 30.66 | 1.172 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | joint | 0.05 | 8.498e-18 | 0.05667 | 0.01258 | 0.365 | 0.1212 | 24.66 | 2.395 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.06667 | 0.03253 | 0.08167 | 0.03753 | 9.553 | 0.7025 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.06333 | 0.01258 | 0.5417 | 0.06788 | 3.136 | 0.5651 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.05333 | 0.02517 | 0.05333 | 0.02517 | 8.509 | 0.7676 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.05833 | 0.01443 | 0.05833 | 0.01443 | 23.13 | 0.9139 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.07667 | 0.01041 | 0.07667 | 0.01041 | 18.94 | 1.136 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.06667 | 0.02466 | 0.06667 | 0.02466 | 9.219 | 0.567 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.06 | 0.01 | 0.06 | 0.01 | 2.76 | 0.5454 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.06667 | 0.01528 | 0.5017 | 0.03175 | 7.503 | 0.3831 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.06333 | 0.03215 | 0.04167 | 0.02566 | 15.63 | 1.207 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.055 | 0.005 | 0.34 | 0.03905 | 13.92 | 0.7065 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.065 | 0.00866 | 0.075 | 0.015 | 8.891 | 0.5429 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.06333 | 0.03512 | 0.57 | 0.035 | 0.9435 | 0.2434 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 8.126 | 0.5363 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 8.126 | 0.5363 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 8.126 | 0.5363 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 8.126 | 0.5363 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 8.126 | 0.5363 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 8.126 | 0.5363 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 8.126 | 0.5363 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.05167 | 0.01528 | 0.05833 | 0.01041 | 8.126 | 0.5363 | 3 | 0 | 0 | 0 | 0 |
| unblocked | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| unblocked | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| unblocked | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.065 | 0.02291 | 0.065 | 0.02291 | 8.6 | 0.7967 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.07833 | 0.05008 | 0.07833 | 0.05008 | 15.71 | 1.543 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.08167 | 0.05204 | 0.08167 | 0.05204 | 15.11 | 1.508 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.08667 | 0.06658 | 0.08667 | 0.06658 | 8.931 | 1.296 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.05667 | 0.02754 | 0.05667 | 0.02754 | 0.6811 | 0.06023 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.05667 | 0.0401 | 0.16 | 0.07263 | 9.39 | 1.169 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | independent | 0.05 | 8.498e-18 | 0.04833 | 0.03819 | 0.02667 | 0.02566 | 31.79 | 2.784 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | joint | 0.05 | 8.498e-18 | 0.06667 | 0.05923 | 0.07333 | 0.08386 | 29.59 | 3.234 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.05167 | 0.03753 | 0.06333 | 0.04752 | 9.933 | 1.154 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.08167 | 0.04252 | 0.3317 | 0.06526 | 1.58 | 0.1419 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.05167 | 0.04041 | 0.05167 | 0.04041 | 9.277 | 1.272 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.055 | 0.05766 | 0.055 | 0.05766 | 24.3 | 2.704 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.05833 | 0.06449 | 0.05833 | 0.06449 | 23.18 | 3.078 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.05667 | 0.03014 | 0.05667 | 0.03014 | 9.5 | 1.112 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.065 | 0.03606 | 0.065 | 0.03606 | 1.342 | 0.1805 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.07167 | 0.02021 | 0.2217 | 0.04481 | 8.268 | 0.4706 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.07 | 0.03279 | 0.04167 | 0.02021 | 15.34 | 0.9168 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.075 | 0.035 | 0.105 | 0.04093 | 14.81 | 0.8283 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.06833 | 0.03512 | 0.08 | 0.04272 | 8.763 | 0.7542 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.06833 | 0.06007 | 0.3467 | 0.08145 | 0.6047 | 0.04002 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 8.111 | 0.6501 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 8.111 | 0.6501 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 8.111 | 0.6501 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 8.111 | 0.6501 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 8.111 | 0.6501 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 8.111 | 0.6501 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 8.111 | 0.6501 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.055 | 0.02291 | 0.05167 | 0.01258 | 8.111 | 0.6501 | 3 | 0 | 0 | 0 | 0 |
| uniform_wide | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| uniform_wide | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| uniform_wide | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | circuit | bank0:0-1 | conditional_max | 0.05 | 8.498e-18 | 0.08833 | 0.05686 | 0.08833 | 0.05686 | 8.782 | 1.027 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | independent | 0.05 | 8.498e-18 | 0.07333 | 0.06028 | 0.07333 | 0.06028 | 15.89 | 1.981 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | joint | 0.05 | 8.498e-18 | 0.07667 | 0.06506 | 0.07667 | 0.06506 | 15.57 | 2.107 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | marginal_max | 0.05 | 8.498e-18 | 0.07333 | 0.05058 | 0.07333 | 0.05058 | 9.209 | 1.267 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | bank0:0-1 | relational_max | 0.05 | 8.498e-18 | 0.03167 | 0.03014 | 0.03167 | 0.03014 | 0.4226 | 0.3752 | 3 | 0.3333 | 0 | 0 | 0 |
| within_capacity | circuit | full | conditional_max | 0.05 | 8.498e-18 | 0.04167 | 0.03403 | 0.08667 | 0.03547 | 9.714 | 1.134 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | independent | 0.05 | 8.498e-18 | 0.07167 | 0.07286 | 0.04667 | 0.05107 | 31.34 | 3.664 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | joint | 0.05 | 8.498e-18 | 0.06167 | 0.05923 | 0.05333 | 0.04311 | 30.11 | 2.948 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | marginal_max | 0.05 | 8.498e-18 | 0.05 | 0.04093 | 0.06667 | 0.05838 | 9.97 | 1.12 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | full | relational_max | 0.05 | 8.498e-18 | 0.05667 | 0.01607 | 0.2367 | 0.08401 | 1.162 | 0.6007 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.05008 | 0.05833 | 0.05008 | 9.387 | 1.153 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | independent | 0.05 | 8.498e-18 | 0.05 | 0.05679 | 0.05 | 0.05679 | 24.06 | 2.502 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | joint | 0.05 | 8.498e-18 | 0.05167 | 0.04368 | 0.05167 | 0.04368 | 23.3 | 1.956 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | marginal_max | 0.05 | 8.498e-18 | 0.06 | 0.05568 | 0.06 | 0.05568 | 9.716 | 1.229 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random1:0 | relational_max | 0.05 | 8.498e-18 | 0.05667 | 0.02021 | 0.05667 | 0.02021 | 0.9329 | 0.3595 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.01528 | 0.095 | 0.01 | 8.901 | 0.4931 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | independent | 0.05 | 8.498e-18 | 0.07167 | 0.04726 | 0.05 | 0.03122 | 15.33 | 1.1 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | joint | 0.05 | 8.498e-18 | 0.07 | 0.04924 | 0.05833 | 0.01756 | 15.33 | 1.101 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | marginal_max | 0.05 | 8.498e-18 | 0.05667 | 0.01756 | 0.08 | 0.02784 | 8.852 | 0.4309 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random2:1-2 | relational_max | 0.05 | 8.498e-18 | 0.05 | 0.035 | 0.125 | 0.1612 | 0.2333 | 0.4039 | 3 | 0.6667 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 8.188 | 0.8203 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | independent | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 8.188 | 0.8203 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | joint | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 8.188 | 0.8203 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | marginal_max | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 8.188 | 0.8203 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | random3:1-2-3 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | conditional_max | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 8.188 | 0.8203 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | independent | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 8.188 | 0.8203 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | joint | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 8.188 | 0.8203 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | marginal_max | 0.05 | 8.498e-18 | 0.05833 | 0.02754 | 0.065 | 0.02598 | 8.188 | 0.8203 | 3 | 0 | 0 | 0 | 0 |
| within_capacity | circuit | single_sensor_0 | relational_max | 0.05 | 8.498e-18 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| within_capacity | known_law_oracle | bank0:0-1 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | bank0:0-1 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | full | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random1:0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random2:1-2 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | random3:1-2-3 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | conditional_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | independent | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | joint | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | marginal_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |
| within_capacity | known_law_oracle | single_sensor_0 | relational_max | — | — | — | — | — | — | — | — | 3 | — | — | — | — |

## Accuracy against the known law

| variant | method_name | mask | oracle_conditional_mae_mean | oracle_conditional_mae_std | oracle_conditional_bias_mean | oracle_conditional_bias_std | oracle_conditional_corr_mean | oracle_conditional_corr_std | oracle_marginal_mae_mean | oracle_marginal_mae_std | oracle_R_mae_mean | oracle_R_mae_std | oracle_R_corr_mean | oracle_R_corr_std | oracle_log_px_mae_mean | oracle_log_px_mae_std | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocked | circuit | bank0:0-1 | 2.237 | 0.1512 | 2.094 | 0.1399 | 0.3551 | 0.0872 | 0.4137 | 0.04933 | 2.1 | 0.136 | 0.211 | 0.2175 | 2.378 | 0.1736 | 3 |
| blocked | circuit | full | 3.163 | 0.02987 | 0.8601 | 0.2466 | 0.4363 | 0.02775 | 0.3865 | 0.02304 | 3.075 | 0.02843 | 0.4719 | 0.06657 | 6.615 | 0.1855 | 3 |
| blocked | circuit | random1:0 | 2.59 | 0.1766 | 2.439 | 0.1918 | 0.2692 | 0.06895 | 0.3918 | 0.02972 | 2.464 | 0.1486 | 0.164 | 0.158 | 4.671 | 0.3635 | 3 |
| blocked | circuit | random2:1-2 | 3.508 | 0.3086 | -1.228 | 0.2731 | 0.5782 | 0.04926 | 0.3825 | 0.01197 | 3.487 | 0.3023 | 0.5293 | 0.06344 | 3.532 | 0.3198 | 3 |
| blocked | circuit | random3:1-2-3 | 0.3706 | 0.004263 | 0.1477 | 0.04973 | 0.945 | 0.002403 | 0.3706 | 0.004263 | 0 | 0 | — | — | 0.3706 | 0.004263 | 3 |
| blocked | circuit | single_sensor_0 | 0.3706 | 0.004263 | 0.1477 | 0.04973 | 0.945 | 0.002403 | 0.3706 | 0.004263 | 0 | 0 | — | — | 0.3706 | 0.004263 | 3 |
| conditional | circuit | bank0:0-1 | 2.112 | 0.006242 | 1.958 | 0.03386 | 0.3931 | 0.04698 | 0.4021 | 0.02022 | 2.002 | 0.009283 | 0.3411 | 0.03645 | 2.221 | 0.009659 | 3 |
| conditional | circuit | full | 3.063 | 0.1283 | 0.7746 | 0.1043 | 0.4696 | 0.0516 | 0.3762 | 0.01622 | 2.981 | 0.1274 | 0.5026 | 0.02751 | 6.461 | 0.3715 | 3 |
| conditional | circuit | random1:0 | 2.464 | 0.06227 | 2.311 | 0.0834 | 0.3149 | 0.07099 | 0.3822 | 0.02217 | 2.361 | 0.04362 | 0.3164 | 0.0577 | 4.418 | 0.1356 | 3 |
| conditional | circuit | random2:1-2 | 3.527 | 0.3022 | -1.25 | 0.2362 | 0.5689 | 0.05329 | 0.3856 | 0.03877 | 3.505 | 0.3006 | 0.5163 | 0.08501 | 3.554 | 0.3118 | 3 |
| conditional | circuit | random3:1-2-3 | 0.3583 | 0.02729 | 0.1364 | 0.0281 | 0.9508 | 0.006036 | 0.3583 | 0.02729 | 0 | 0 | — | — | 0.3583 | 0.02729 | 3 |
| conditional | circuit | single_sensor_0 | 0.3583 | 0.02729 | 0.1364 | 0.0281 | 0.9508 | 0.006036 | 0.3583 | 0.02729 | 0 | 0 | — | — | 0.3583 | 0.02729 | 3 |
| conditional_nll_selection | circuit | bank0:0-1 | 2.112 | 0.006242 | 1.958 | 0.03386 | 0.3931 | 0.04698 | 0.4021 | 0.02022 | 2.002 | 0.009283 | 0.3411 | 0.03645 | 2.221 | 0.009659 | 3 |
| conditional_nll_selection | circuit | full | 3.063 | 0.1283 | 0.7746 | 0.1043 | 0.4696 | 0.0516 | 0.3762 | 0.01622 | 2.981 | 0.1274 | 0.5026 | 0.02751 | 6.461 | 0.3715 | 3 |
| conditional_nll_selection | circuit | random1:0 | 2.464 | 0.06227 | 2.311 | 0.0834 | 0.3149 | 0.07099 | 0.3822 | 0.02217 | 2.361 | 0.04362 | 0.3164 | 0.0577 | 4.418 | 0.1356 | 3 |
| conditional_nll_selection | circuit | random2:1-2 | 3.527 | 0.3022 | -1.25 | 0.2362 | 0.5689 | 0.05329 | 0.3856 | 0.03877 | 3.505 | 0.3006 | 0.5163 | 0.08501 | 3.554 | 0.3118 | 3 |
| conditional_nll_selection | circuit | random3:1-2-3 | 0.3583 | 0.02729 | 0.1364 | 0.0281 | 0.9508 | 0.006036 | 0.3583 | 0.02729 | 0 | 0 | — | — | 0.3583 | 0.02729 | 3 |
| conditional_nll_selection | circuit | single_sensor_0 | 0.3583 | 0.02729 | 0.1364 | 0.0281 | 0.9508 | 0.006036 | 0.3583 | 0.02729 | 0 | 0 | — | — | 0.3583 | 0.02729 | 3 |
| interface | circuit | bank0:0-1 | 2.076 | 0.03664 | 1.902 | 0.01038 | 0.3799 | 0.06369 | 0.3686 | 0.03864 | 1.965 | 0.04234 | 0.3508 | 0.04585 | 2.185 | 0.03524 | 3 |
| interface | circuit | full | 3.034 | 0.03685 | 0.7251 | 0.156 | 0.4656 | 0.02398 | 0.3604 | 0.02256 | 2.957 | 0.03701 | 0.494 | 0.05588 | 6.399 | 0.1667 | 3 |
| interface | circuit | random1:0 | 2.441 | 0.07282 | 2.285 | 0.09927 | 0.3057 | 0.06318 | 0.3606 | 0.01901 | 2.338 | 0.06667 | 0.3143 | 0.0532 | 4.395 | 0.1461 | 3 |
| interface | circuit | random2:1-2 | 3.569 | 0.2 | -1.314 | 0.2454 | 0.5421 | 0.001058 | 0.3706 | 0.02125 | 3.548 | 0.1949 | 0.4789 | 0.02447 | 3.596 | 0.2098 | 3 |
| interface | circuit | random3:1-2-3 | 0.36 | 0.03396 | 0.127 | 0.02464 | 0.9506 | 0.008568 | 0.36 | 0.03396 | 0 | 0 | — | — | 0.36 | 0.03396 | 3 |
| interface | circuit | single_sensor_0 | 0.36 | 0.03396 | 0.127 | 0.02464 | 0.9506 | 0.008568 | 0.36 | 0.03396 | 0 | 0 | — | — | 0.36 | 0.03396 | 3 |
| interface_conditional | circuit | bank0:0-1 | 2.065 | 0.04591 | 1.897 | 0.03049 | 0.4 | 0.05869 | 0.3672 | 0.02909 | 1.958 | 0.0493 | 0.3653 | 0.03744 | 2.17 | 0.04438 | 3 |
| interface_conditional | circuit | full | 3 | 0.04746 | 0.7046 | 0.1643 | 0.4859 | 0.02192 | 0.3554 | 0.01668 | 2.921 | 0.04761 | 0.5263 | 0.02773 | 6.346 | 0.207 | 3 |
| interface_conditional | circuit | random1:0 | 2.405 | 0.06152 | 2.243 | 0.08865 | 0.3155 | 0.05412 | 0.3603 | 0.01639 | 2.302 | 0.05997 | 0.3194 | 0.04487 | 4.314 | 0.1185 | 3 |
| interface_conditional | circuit | random2:1-2 | 3.459 | 0.2029 | -1.279 | 0.2726 | 0.5877 | 0.02885 | 0.3563 | 0.01915 | 3.432 | 0.1945 | 0.5328 | 0.05755 | 3.488 | 0.21 | 3 |
| interface_conditional | circuit | random3:1-2-3 | 0.341 | 0.02352 | 0.1184 | 0.04522 | 0.9515 | 0.0085 | 0.341 | 0.02352 | 0 | 0 | — | — | 0.341 | 0.02352 | 3 |
| interface_conditional | circuit | single_sensor_0 | 0.341 | 0.02352 | 0.1184 | 0.04522 | 0.9515 | 0.0085 | 0.341 | 0.02352 | 0 | 0 | — | — | 0.341 | 0.02352 | 3 |
| mixture | circuit | bank0:0-1 | 1.811 | 0.02808 | 1.597 | 0.02349 | 0.4475 | 0.1049 | 0.3826 | 0.03147 | 1.682 | 0.02038 | 0.4941 | 0.05173 | 1.951 | 0.04039 | 3 |
| mixture | circuit | full | 2.704 | 0.116 | 0.483 | 0.1049 | 0.5928 | 0.01709 | 0.3852 | 0.02755 | 2.619 | 0.1262 | 0.626 | 0.03227 | 5.737 | 0.3373 | 3 |
| mixture | circuit | random1:0 | 2.086 | 0.009929 | 1.903 | 0.006465 | 0.4128 | 0.06846 | 0.3884 | 0.03041 | 1.971 | 0.03568 | 0.4924 | 0.03499 | 3.723 | 0.04423 | 3 |
| mixture | circuit | random2:1-2 | 3.147 | 0.2096 | -1.226 | 0.1908 | 0.6949 | 0.03984 | 0.3828 | 0.02081 | 3.119 | 0.1969 | 0.6748 | 0.05039 | 3.181 | 0.2198 | 3 |
| mixture | circuit | random3:1-2-3 | 0.3757 | 0.01913 | 0.1656 | 0.04574 | 0.9415 | 0.005864 | 0.3757 | 0.01913 | 0 | 0 | — | — | 0.3757 | 0.01913 | 3 |
| mixture | circuit | single_sensor_0 | 0.3757 | 0.01913 | 0.1656 | 0.04574 | 0.9415 | 0.005864 | 0.3757 | 0.01913 | 0 | 0 | — | — | 0.3757 | 0.01913 | 3 |
| random | circuit | bank0:0-1 | 2.165 | 0.08565 | 2.019 | 0.04732 | 0.3957 | 0.06324 | 0.5948 | 0.1029 | 1.902 | 0.1136 | 0.3327 | 0.1292 | 2.443 | 0.08819 | 3 |
| random | circuit | full | 2.937 | 0.07369 | 0.9474 | 0.2251 | 0.5718 | 0.04629 | 0.5748 | 0.02745 | 2.745 | 0.06609 | 0.5948 | 0.0233 | 6.235 | 0.1385 | 3 |
| random | circuit | random1:0 | 2.47 | 0.1273 | 2.316 | 0.1324 | 0.3341 | 0.05567 | 0.5795 | 0.06201 | 2.214 | 0.1111 | 0.3088 | 0.08511 | 4.671 | 0.2277 | 3 |
| random | circuit | random2:1-2 | 3.302 | 0.3369 | -0.9802 | 0.3568 | 0.6171 | 0.07095 | 0.5812 | 0.02068 | 3.237 | 0.3587 | 0.5641 | 0.1076 | 3.387 | 0.3102 | 3 |
| random | circuit | random3:1-2-3 | 0.5605 | 0.07754 | 0.2638 | 0.08448 | 0.8857 | 0.02135 | 0.5605 | 0.07754 | 0 | 0 | — | — | 0.5605 | 0.07754 | 3 |
| random | circuit | single_sensor_0 | 0.5605 | 0.07754 | 0.2638 | 0.08448 | 0.8857 | 0.02135 | 0.5605 | 0.07754 | 0 | 0 | — | — | 0.5605 | 0.07754 | 3 |
| unblocked | circuit | bank0:0-1 | 1.372 | 0.06895 | 1.13 | 0.06868 | 0.6062 | 0.02495 | 0.4898 | 0.06071 | 1.189 | 0.1 | 0.6796 | 0.05069 | 1.602 | 0.01317 | 3 |
| unblocked | circuit | full | 2.08 | 0.07763 | 0.4878 | 0.1548 | 0.7842 | 0.01918 | 0.4841 | 0.0506 | 1.938 | 0.109 | 0.7864 | 0.02776 | 4.51 | 0.1385 | 3 |
| unblocked | circuit | random1:0 | 1.508 | 0.02249 | 1.273 | 0.02354 | 0.5797 | 0.05433 | 0.4889 | 0.05554 | 1.348 | 0.03643 | 0.6751 | 0.02811 | 2.765 | 0.05028 | 3 |
| unblocked | circuit | random2:1-2 | 2.415 | 0.1779 | -0.5652 | 0.3239 | 0.7939 | 0.01547 | 0.4709 | 0.04645 | 2.343 | 0.1921 | 0.7696 | 0.02533 | 2.523 | 0.1467 | 3 |
| unblocked | circuit | random3:1-2-3 | 0.4696 | 0.05573 | 0.219 | 0.06495 | 0.9036 | 0.02333 | 0.4696 | 0.05573 | 0 | 0 | — | — | 0.4696 | 0.05573 | 3 |
| unblocked | circuit | single_sensor_0 | 0.4696 | 0.05573 | 0.219 | 0.06495 | 0.9036 | 0.02333 | 0.4696 | 0.05573 | 0 | 0 | — | — | 0.4696 | 0.05573 | 3 |
| uniform_wide | circuit | bank0:0-1 | 2.11 | 0.05794 | 1.941 | 0.09088 | 0.3972 | 0.0457 | 0.3738 | 0.02507 | 1.99 | 0.06112 | 0.3305 | 0.05757 | 2.231 | 0.05605 | 3 |
| uniform_wide | circuit | full | 3.058 | 0.105 | 0.7465 | 0.1685 | 0.4571 | 0.0288 | 0.3579 | 0.0158 | 2.978 | 0.102 | 0.4923 | 0.04977 | 6.395 | 0.384 | 3 |
| uniform_wide | circuit | random1:0 | 2.451 | 0.04558 | 2.285 | 0.04315 | 0.3189 | 0.04203 | 0.3589 | 0.01148 | 2.343 | 0.05704 | 0.31 | 0.008915 | 4.398 | 0.06856 | 3 |
| uniform_wide | circuit | random2:1-2 | 3.47 | 0.3567 | -1.277 | 0.3061 | 0.5895 | 0.06973 | 0.3492 | 0.03033 | 3.446 | 0.3542 | 0.5457 | 0.09221 | 3.496 | 0.3591 | 3 |
| uniform_wide | circuit | random3:1-2-3 | 0.3552 | 0.04237 | 0.1227 | 0.05386 | 0.952 | 0.01296 | 0.3552 | 0.04237 | 0 | 0 | — | — | 0.3552 | 0.04237 | 3 |
| uniform_wide | circuit | single_sensor_0 | 0.3552 | 0.04237 | 0.1227 | 0.05386 | 0.952 | 0.01296 | 0.3552 | 0.04237 | 0 | 0 | — | — | 0.3552 | 0.04237 | 3 |
| within_capacity | circuit | bank0:0-1 | 2.245 | 0.1349 | 2.07 | 0.1787 | 0.3446 | 0.04466 | 0.3855 | 0.04438 | 2.11 | 0.1406 | 0.1471 | 0.2355 | 2.378 | 0.1351 | 3 |
| within_capacity | circuit | full | 3.184 | 0.1714 | 0.7828 | 0.1161 | 0.3648 | 0.09859 | 0.3835 | 0.03001 | 3.093 | 0.1526 | 0.3656 | 0.1038 | 6.776 | 0.4655 | 3 |
| within_capacity | circuit | random1:0 | 2.572 | 0.1144 | 2.417 | 0.1488 | 0.2839 | 0.0687 | 0.3894 | 0.0454 | 2.439 | 0.08785 | 0.2509 | 0.06471 | 4.701 | 0.2839 | 3 |
| within_capacity | circuit | random2:1-2 | 3.801 | 0.398 | -1.272 | 0.2431 | 0.4255 | 0.09614 | 0.3772 | 0.03051 | 3.777 | 0.3884 | 0.07251 | 0.3298 | 3.831 | 0.404 | 3 |
| within_capacity | circuit | random3:1-2-3 | 0.3658 | 0.01828 | 0.1362 | 0.03365 | 0.9518 | 0.00737 | 0.3658 | 0.01828 | 0 | 0 | — | — | 0.3658 | 0.01828 | 3 |
| within_capacity | circuit | single_sensor_0 | 0.3658 | 0.01828 | 0.1362 | 0.03365 | 0.9518 | 0.00737 | 0.3658 | 0.01828 | 0 | 0 | — | — | 0.3658 | 0.01828 | 3 |

## Cost

| variant | method_name | mask | query_s_mean | query_s_std | warmup_s_mean | warmup_s_std | cost_passes_mean | cost_passes_std | cost_node_evaluations_mean | cost_node_evaluations_std | cost_n_masks_mean | cost_n_masks_std | cost_output_elements_mean | cost_output_elements_std | cost_peak_boundary_bytes_mean | cost_peak_boundary_bytes_std | n_observed_mean | n_observed_std | oracle_max_error_mean | oracle_max_error_std | n_seeds | mask_shared_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocked | circuit | bank0:0-1 | 0.006556 | 7.472e-05 | 0.004384 | 8.128e-05 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 2 | 0 | 2.066e-06 | 2.753e-07 | 3 | 1 |
| blocked | circuit | full | 0.006705 | 0.0001481 | 0.004565 | 9.444e-05 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 4 | 0 | 4.133e-06 | 5.506e-07 | 3 | 1 |
| blocked | circuit | random1:0 | 0.006635 | 0.0001103 | 0.004415 | 0.0001538 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 3 | 0 | 2.066e-06 | 2.753e-07 | 3 | 1 |
| blocked | circuit | random2:1-2 | 0.006596 | 1.994e-05 | 0.004436 | 1.699e-05 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 2 | 0 | 1.748e-06 | 2.753e-07 | 3 | 1 |
| blocked | circuit | random3:1-2-3 | 0.006631 | 6.65e-05 | 0.00442 | 1.619e-05 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 1 | 0 | 7.947e-07 | 2.753e-07 | 3 | 1 |
| blocked | circuit | single_sensor_0 | 0.006585 | 7.329e-05 | 0.004376 | 7.559e-05 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 1 | 0 | 7.947e-07 | 2.753e-07 | 3 | 1 |
| conditional | circuit | bank0:0-1 | 0.00672 | 0.0002566 | 0.004124 | 0.000181 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 2 | 0 | 1.987e-06 | 3.642e-07 | 3 | 1 |
| conditional | circuit | full | 0.006915 | 0.0001283 | 0.004311 | 0.0002564 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 4 | 0 | 4.45e-06 | 1.101e-06 | 3 | 1 |
| conditional | circuit | random1:0 | 0.009429 | 0.004622 | 0.004442 | 0.0004903 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 3 | 0 | 3.338e-06 | 4.768e-07 | 3 | 1 |
| conditional | circuit | random2:1-2 | 0.006992 | 0.0007773 | 0.004219 | 0.0003353 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 2 | 0 | 1.907e-06 | 0 | 3 | 1 |
| conditional | circuit | random3:1-2-3 | 0.006728 | 0.000121 | 0.004399 | 0.0005788 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 1 | 0 | 1.033e-06 | 3.642e-07 | 3 | 1 |
| conditional | circuit | single_sensor_0 | 0.00676 | 0.0002908 | 0.004123 | 6.769e-05 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 1 | 0 | 1.033e-06 | 3.642e-07 | 3 | 1 |
| conditional_nll_selection | circuit | bank0:0-1 | 0.007693 | 0.001759 | 0.004193 | 0.0001657 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 2 | 0 | 1.987e-06 | 3.642e-07 | 3 | 1 |
| conditional_nll_selection | circuit | full | 0.007185 | 0.0001343 | 0.004923 | 0.0005663 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 4 | 0 | 4.45e-06 | 1.101e-06 | 3 | 1 |
| conditional_nll_selection | circuit | random1:0 | 0.006929 | 0.0002237 | 0.004161 | 0.0001229 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 3 | 0 | 3.338e-06 | 4.768e-07 | 3 | 1 |
| conditional_nll_selection | circuit | random2:1-2 | 0.006753 | 0.0002884 | 0.00414 | 5.544e-05 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 2 | 0 | 1.907e-06 | 0 | 3 | 1 |
| conditional_nll_selection | circuit | random3:1-2-3 | 0.008044 | 0.000858 | 0.00446 | 0.0002733 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 1 | 0 | 1.033e-06 | 3.642e-07 | 3 | 1 |
| conditional_nll_selection | circuit | single_sensor_0 | 0.006699 | 0.0004038 | 0.004186 | 0.0002806 | 3 | 0 | 1.8e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 1 | 0 | 1.033e-06 | 3.642e-07 | 3 | 1 |
| interface | circuit | bank0:0-1 | 0.01401 | 0.0005514 | 0.009053 | 0.0002341 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 2 | 0 | 1.907e-06 | 4.768e-07 | 3 | 1 |
| interface | circuit | full | 0.01381 | 0.0003843 | 0.009159 | 7.353e-05 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 4 | 0 | 5.086e-06 | 7.284e-07 | 3 | 1 |
| interface | circuit | random1:0 | 0.01386 | 0.0001459 | 0.008882 | 9.332e-05 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 3 | 0 | 3.497e-06 | 1.533e-06 | 3 | 1 |
| interface | circuit | random2:1-2 | 0.01386 | 0.0004944 | 0.008828 | 0.0001446 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 2 | 0 | 2.066e-06 | 2.753e-07 | 3 | 1 |
| interface | circuit | random3:1-2-3 | 0.014 | 0.0006629 | 0.009661 | 0.001177 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 1 | 0 | 9.537e-07 | 0 | 3 | 1 |
| interface | circuit | single_sensor_0 | 0.01391 | 0.0004628 | 0.009166 | 0.0006254 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 1 | 0 | 9.537e-07 | 0 | 3 | 1 |
| interface_conditional | circuit | bank0:0-1 | 0.01423 | 0.0002622 | 0.008356 | 4.768e-05 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 2 | 0 | 3.02e-06 | 1.377e-06 | 3 | 1 |
| interface_conditional | circuit | full | 0.01428 | 0.0001407 | 0.008552 | 0.0002693 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 4 | 0 | 4.45e-06 | 1.101e-06 | 3 | 1 |
| interface_conditional | circuit | random1:0 | 0.01417 | 0.0002037 | 0.008449 | 0.0001796 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 3 | 0 | 3.735e-06 | 1.725e-06 | 3 | 1 |
| interface_conditional | circuit | random2:1-2 | 0.01431 | 0.0003736 | 0.008218 | 0.000141 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 2 | 0 | 1.907e-06 | 4.768e-07 | 3 | 1 |
| interface_conditional | circuit | random3:1-2-3 | 0.01436 | 0.0001716 | 0.008221 | 0.0001115 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 1 | 0 | 6.358e-07 | 2.753e-07 | 3 | 1 |
| interface_conditional | circuit | single_sensor_0 | 0.0143 | 0.0006359 | 0.008426 | 0.0003312 | 3 | 0 | 3.208e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 1 | 0 | 6.358e-07 | 2.753e-07 | 3 | 1 |
| mixture | circuit | bank0:0-1 | 0.006012 | 0.0002196 | 0.003709 | 0.0001354 | 3 | 0 | 1.736e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 2 | 0 | 1.431e-06 | 4.768e-07 | 3 | 1 |
| mixture | circuit | full | 0.006138 | 0.0002887 | 0.003957 | 0.0001352 | 3 | 0 | 1.736e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 4 | 0 | 3.497e-06 | 5.506e-07 | 3 | 1 |
| mixture | circuit | random1:0 | 0.006043 | 0.0001808 | 0.003782 | 0.0002098 | 3 | 0 | 1.736e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 3 | 0 | 2.225e-06 | 2.753e-07 | 3 | 1 |
| mixture | circuit | random2:1-2 | 0.006213 | 0.000225 | 0.003765 | 0.0001782 | 3 | 0 | 1.736e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 2 | 0 | 1.351e-06 | 4.963e-07 | 3 | 1 |
| mixture | circuit | random3:1-2-3 | 0.005959 | 0.0002323 | 0.003772 | 0.0001433 | 3 | 0 | 1.736e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 1 | 0 | 4.768e-07 | 0 | 3 | 1 |
| mixture | circuit | single_sensor_0 | 0.005858 | 9.523e-05 | 0.003684 | 0.0001186 | 3 | 0 | 1.736e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 1 | 0 | 4.768e-07 | 0 | 3 | 1 |
| random | circuit | bank0:0-1 | 0.009464 | 0.0005207 | 0.002163 | 0.0001195 | 5 | 0 | 7.22e+05 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 1 |
| random | circuit | full | 0.02031 | 0.0005799 | 0.003076 | 0.0001757 | 9 | 0 | 1.3e+06 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 | 1 |
| random | circuit | random1:0 | 0.01254 | 0.0007466 | 0.002563 | 2.675e-05 | 7 | 0 | 1.011e+06 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 1 |
| random | circuit | random2:1-2 | 0.009209 | 0.0001007 | 0.002125 | 2.77e-05 | 5 | 0 | 7.22e+05 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 1 |
| random | circuit | random3:1-2-3 | 0.006304 | 0.0001484 | 0.001748 | 4.29e-05 | 3 | 0 | 4.332e+05 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 1 |
| random | circuit | single_sensor_0 | 0.006591 | 0.0004096 | 0.001736 | 7.569e-05 | 3 | 0 | 4.332e+05 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 1 |
| unblocked | circuit | bank0:0-1 | 0.01032 | 0.0005423 | 0.002268 | 4.119e-05 | 5 | 0 | 7.22e+05 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 1 |
| unblocked | circuit | full | 0.02206 | 0.0001985 | 0.003319 | 8.567e-05 | 9 | 0 | 1.3e+06 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 | 1 |
| unblocked | circuit | random1:0 | 0.01483 | 0.0004544 | 0.002813 | 4.711e-05 | 7 | 0 | 1.011e+06 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 1 |
| unblocked | circuit | random2:1-2 | 0.01005 | 0.0002636 | 0.0023 | 3.337e-05 | 5 | 0 | 7.22e+05 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 1 |
| unblocked | circuit | random3:1-2-3 | 0.006672 | 0.0002999 | 0.00185 | 1.354e-05 | 3 | 0 | 4.332e+05 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 1 |
| unblocked | circuit | single_sensor_0 | 0.006837 | 0.0001941 | 0.001902 | 0.0001455 | 3 | 0 | 4.332e+05 | 0 | 1 | 0 | 1600 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 1 |
| uniform_wide | circuit | bank0:0-1 | 0.0251 | 0.0007647 | 0.01293 | 0.0004702 | 3 | 0 | 5.896e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 2 | 0 | 2.384e-06 | 4.768e-07 | 3 | 1 |
| uniform_wide | circuit | full | 0.02403 | 0.0006649 | 0.0128 | 0.0006605 | 3 | 0 | 5.896e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 4 | 0 | 5.563e-06 | 9.926e-07 | 3 | 1 |
| uniform_wide | circuit | random1:0 | 0.02456 | 0.0006177 | 0.0124 | 6.44e-05 | 3 | 0 | 5.896e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 3 | 0 | 3.656e-06 | 2.753e-07 | 3 | 1 |
| uniform_wide | circuit | random2:1-2 | 0.024 | 0.001105 | 0.0126 | 0.0001081 | 3 | 0 | 5.896e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 2 | 0 | 2.225e-06 | 7.284e-07 | 3 | 1 |
| uniform_wide | circuit | random3:1-2-3 | 0.02419 | 0.001544 | 0.01276 | 0.0007237 | 3 | 0 | 5.896e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 1 | 0 | 8.742e-07 | 1.377e-07 | 3 | 1 |
| uniform_wide | circuit | single_sensor_0 | 0.02407 | 0.001106 | 0.01262 | 0.0003826 | 3 | 0 | 5.896e+05 | 0 | 1 | 0 | 1600 | 0 | 5.12e+04 | 0 | 1 | 0 | 8.742e-07 | 1.377e-07 | 3 | 1 |
| within_capacity | circuit | bank0:0-1 | 0.01604 | 0.0005969 | 0.008094 | 0.0001288 | 3 | 0 | 4.488e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 2 | 0 | 2.066e-06 | 7.284e-07 | 3 | 1 |
| within_capacity | circuit | full | 0.04403 | 0.04566 | 0.008744 | 0.000499 | 3 | 0 | 4.488e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 4 | 0 | 3.974e-06 | 2.753e-07 | 3 | 1 |
| within_capacity | circuit | random1:0 | 0.01639 | 0.001542 | 0.008237 | 7.2e-06 | 3 | 0 | 4.488e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 3 | 0 | 3.02e-06 | 2.753e-07 | 3 | 1 |
| within_capacity | circuit | random2:1-2 | 0.01607 | 0.001223 | 0.00842 | 0.0005984 | 3 | 0 | 4.488e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 2 | 0 | 2.225e-06 | 5.506e-07 | 3 | 1 |
| within_capacity | circuit | random3:1-2-3 | 0.01594 | 0.001274 | 0.008164 | 0.0003676 | 3 | 0 | 4.488e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 1 | 0 | 9.537e-07 | 0 | 3 | 1 |
| within_capacity | circuit | single_sensor_0 | 0.01591 | 0.001002 | 0.008101 | 7.945e-05 | 3 | 0 | 4.488e+05 | 0 | 1 | 0 | 1600 | 0 | 2.56e+04 | 0 | 1 | 0 | 9.537e-07 | 0 | 3 | 1 |

## Marginal-shortcut audit

| variant | shortcut_auroc_mean | shortcut_auroc_std | shortcut_n_dev_mean | shortcut_n_dev_std | shortcut_n_eval_mean | shortcut_n_eval_std | n_seeds |
|---|---|---|---|---|---|---|---|
| blocked | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| conditional | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| conditional_nll_selection | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| interface | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| interface_conditional | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| mixture | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| random | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| unblocked | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| uniform_wide | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |
| within_capacity | 0.4815 | 0.02226 | 400 | 0 | 400 | 0 | 3 |

## Paired comparisons (circuit − comparator)

_no paired artifacts found_

## Reading notes

- Intervals resample ENGINES; repeated training seeds are a separate axis and are reported as `n_seeds` / `_std`, never folded into an interval.
- `loc_ap` excludes windows where every observed channel is a target (`loc_n_all_observed_are_targets`) — average precision is undefined there — and windows with no observed target (`loc_n_no_observed_target`).
- `loc_n_ambiguous` counts tied top scores; no winner is forced, so a large ambiguity count with a high `loc_unique_top1_accuracy` is a small sample, not a good method.
- An infinite threshold is the honest answer to too few independent calibration objects. `trial_infinite_threshold_frac` says how often that happened; it is never clipped to a finite value.
- Query seconds compare THIS implementation of each method; node-row evaluations compare the algorithms and are a work proxy, not FLOPs.
- Masks whose `mechanism` is not `externally_fixed_shared` are stress tests: the conservative threshold's exchangeability argument does not extend to them.

