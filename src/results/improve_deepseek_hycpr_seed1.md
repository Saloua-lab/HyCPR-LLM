# Improvement experiments, deepseek (selection on validation, numbers on test)

## LDOS-WH

Table 4: deepseek 0.808 / 1.097; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.815 / 1.111 | 0.803 / 1.071 |  |
| v0+hybrid **chosen** | 0.792 / 1.024 | 0.751 / 0.974 | FM w=0.5 |

chosen vs best baseline: MAE +0.006 [-0.029, +0.042], RMSE +0.027 [-0.014, +0.069]

chosen vs Table 4 LLM: MAE -0.057 [-0.095, -0.020] *, RMSE -0.123 [-0.166, -0.080] *

## Depaul-WH

Table 4: deepseek 1.154 / 1.375; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.104 / 1.342 | 1.159 / 1.374 |  |
| v0+hybrid **chosen** | 0.797 / 1.027 | 0.767 / 0.981 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.028, +0.083] *, RMSE -0.011 [-0.043, +0.021]

chosen vs Table 4 LLM: MAE -0.387 [-0.449, -0.327] *, RMSE -0.393 [-0.465, -0.323] *

## Tijuana-WH

Table 4: deepseek 0.764 / 1.097; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.848 / 1.221 | 0.781 / 1.132 |  |
| v0+hybrid **chosen** | 0.641 / 1.031 | 0.519 / 0.825 | DeepFM w=0.1 |

chosen vs best baseline: MAE +0.022 [+0.012, +0.031] *, RMSE +0.001 [-0.008, +0.010]

chosen vs Table 4 LLM: MAE -0.245 [-0.324, -0.169] *, RMSE -0.272 [-0.367, -0.182] *

## LDOS-H

Table 4: deepseek 0.803 / 1.037; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.790 / 1.061 | 0.785 / 1.013 |  |
| v0+hybrid **chosen** | 0.793 / 1.033 | 0.742 / 0.937 | CAMF_CU w=0.65 |

chosen vs best baseline: MAE -0.003 [-0.037, +0.032], RMSE -0.011 [-0.044, +0.024]

chosen vs Table 4 LLM: MAE -0.062 [-0.096, -0.028] *, RMSE -0.100 [-0.137, -0.064] *

## Depaul-H

Table 4: deepseek 0.799 / 1.118; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.793 / 1.151 | 0.806 / 1.160 |  |
| v0+hybrid **chosen** | 0.763 / 1.016 | 0.741 / 0.975 | NFM w=0.25 |

chosen vs best baseline: MAE +0.028 [+0.002, +0.056] *, RMSE -0.017 [-0.049, +0.016]

chosen vs Table 4 LLM: MAE -0.058 [-0.101, -0.016] *, RMSE -0.142 [-0.197, -0.089] *

## Restaurant-H

Table 4: deepseek 0.675 / 1.108; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.749 / 1.146 | 0.683 / 1.046 |  |
| v0+hybrid **chosen** | 0.647 / 1.018 | 0.532 / 0.826 | DeepFM w=0.25 |

chosen vs best baseline: MAE +0.035 [+0.012, +0.057] *, RMSE +0.001 [-0.020, +0.024]

chosen vs Table 4 LLM: MAE -0.143 [-0.217, -0.074] *, RMSE -0.282 [-0.380, -0.191] *

