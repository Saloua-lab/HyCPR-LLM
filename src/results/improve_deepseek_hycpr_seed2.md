# Improvement experiments, deepseek (selection on validation, numbers on test)

## LDOS-WH

Table 4: deepseek 0.808 / 1.097; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.821 / 1.147 | 0.812 / 1.101 |  |
| v0+hybrid **chosen** | 0.811 / 1.038 | 0.754 / 0.975 | CAMF_CUCI w=0.45 |

chosen vs best baseline: MAE +0.009 [-0.024, +0.043], RMSE +0.027 [-0.012, +0.068]

chosen vs Table 4 LLM: MAE -0.054 [-0.094, -0.016] *, RMSE -0.122 [-0.169, -0.076] *

## Depaul-WH

Table 4: deepseek 1.154 / 1.375; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.100 / 1.344 | 1.154 / 1.375 |  |
| v0+hybrid **chosen** | 0.796 / 1.027 | 0.767 / 0.981 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.027, +0.083] *, RMSE -0.011 [-0.043, +0.021]

chosen vs Table 4 LLM: MAE -0.387 [-0.449, -0.327] *, RMSE -0.393 [-0.465, -0.323] *

## Tijuana-WH

Table 4: deepseek 0.764 / 1.097; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.866 / 1.242 | 0.853 / 1.153 |  |
| v0+hybrid **chosen** | 0.619 / 1.033 | 0.498 / 0.824 | DeepFM w=0.0 |

chosen vs best baseline: MAE +0.000 [+0.000, +0.000], RMSE +0.000 [+0.000, +0.000]

chosen vs Table 4 LLM: MAE -0.267 [-0.353, -0.184] *, RMSE -0.273 [-0.374, -0.177] *

## LDOS-H

Table 4: deepseek 0.803 / 1.037; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.810 / 1.090 | 0.823 / 1.056 |  |
| v0+hybrid **chosen** | 0.801 / 1.035 | 0.753 / 0.950 | FM w=0.55 |

chosen vs best baseline: MAE +0.008 [-0.026, +0.042], RMSE +0.003 [-0.034, +0.041]

chosen vs Table 4 LLM: MAE -0.051 [-0.084, -0.018] *, RMSE -0.087 [-0.123, -0.051] *

## Depaul-H

Table 4: deepseek 0.799 / 1.118; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.754 / 1.085 | 0.814 / 1.146 |  |
| v0+hybrid **chosen** | 0.732 / 0.996 | 0.743 / 0.979 | FM w=0.45 |

chosen vs best baseline: MAE +0.031 [-0.002, +0.064], RMSE -0.012 [-0.052, +0.027]

chosen vs Table 4 LLM: MAE -0.055 [-0.096, -0.015] *, RMSE -0.138 [-0.188, -0.089] *

## Restaurant-H

Table 4: deepseek 0.675 / 1.108; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.772 / 1.216 | 0.661 / 1.031 |  |
| v0+hybrid **chosen** | 0.638 / 1.028 | 0.512 / 0.816 | DeepFM w=0.15 |

chosen vs best baseline: MAE +0.014 [+0.001, +0.028] *, RMSE -0.008 [-0.021, +0.006]

chosen vs Table 4 LLM: MAE -0.163 [-0.243, -0.089] *, RMSE -0.291 [-0.396, -0.194] *

