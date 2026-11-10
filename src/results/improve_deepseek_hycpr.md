# Improvement experiments, deepseek (selection on validation, numbers on test)

## LDOS-WH

Table 4: deepseek 0.808 / 1.097; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.771 / 1.078 | 0.803 / 1.082 |  |
| v0+hybrid **chosen** | 0.780 / 1.010 | 0.754 / 0.979 | CAMF_CUCI w=0.55 |

chosen vs best baseline: MAE +0.009 [-0.025, +0.044], RMSE +0.032 [-0.009, +0.074]

chosen vs Table 4 LLM: MAE -0.054 [-0.091, -0.018] *, RMSE -0.118 [-0.160, -0.076] *

## Depaul-WH

Table 4: deepseek 1.154 / 1.375; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.100 / 1.336 | 1.158 / 1.371 |  |
| v0+hybrid **chosen** | 0.808 / 1.026 | 0.781 / 0.983 | NFM w=0.15 |

chosen vs best baseline: MAE +0.068 [+0.040, +0.097] *, RMSE -0.009 [-0.042, +0.024]

chosen vs Table 4 LLM: MAE -0.373 [-0.432, -0.317] *, RMSE -0.392 [-0.460, -0.325] *

## Tijuana-WH

Table 4: deepseek 0.764 / 1.097; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.903 / 1.283 | 0.911 / 1.222 |  |
| v0+hybrid **chosen** | 0.619 / 1.033 | 0.498 / 0.824 | DeepFM w=0.0 |

chosen vs best baseline: MAE +0.000 [+0.000, +0.000], RMSE +0.000 [+0.000, +0.000]

chosen vs Table 4 LLM: MAE -0.267 [-0.353, -0.184] *, RMSE -0.273 [-0.374, -0.177] *

## LDOS-H

Table 4: deepseek 0.803 / 1.037; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.803 / 1.052 | 0.811 / 1.035 |  |
| v0+hybrid **chosen** | 0.796 / 1.024 | 0.749 / 0.948 | FM w=0.65 |

chosen vs best baseline: MAE +0.004 [-0.033, +0.042], RMSE +0.000 [-0.038, +0.040]

chosen vs Table 4 LLM: MAE -0.054 [-0.088, -0.022] *, RMSE -0.089 [-0.125, -0.055] *

## Depaul-H

Table 4: deepseek 0.799 / 1.118; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.791 / 1.109 | 0.817 / 1.149 |  |
| v0+hybrid **chosen** | 0.734 / 1.011 | 0.721 / 0.982 | CAMF_CU w=0.4 |

chosen vs best baseline: MAE +0.008 [-0.015, +0.032], RMSE -0.010 [-0.037, +0.018]

chosen vs Table 4 LLM: MAE -0.078 [-0.120, -0.037] *, RMSE -0.136 [-0.189, -0.083] *

## Restaurant-H

Table 4: deepseek 0.675 / 1.108; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.691 / 1.114 | 0.659 / 1.055 |  |
| v0+hybrid **chosen** | 0.636 / 1.020 | 0.531 / 0.832 | DeepFM w=0.25 |

chosen vs best baseline: MAE +0.033 [+0.012, +0.055] *, RMSE +0.008 [-0.014, +0.032]

chosen vs Table 4 LLM: MAE -0.144 [-0.215, -0.078] *, RMSE -0.276 [-0.371, -0.185] *

