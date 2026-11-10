# Improvement experiments, eurollm (selection on validation, numbers on test)

## LDOS-WH

Table 4: eurollm 0.832 / 1.109; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.781 / 1.050 | 0.830 / 1.119 |  |
| v0+hybrid **chosen** | 0.775 / 1.001 | 0.775 / 1.013 | FM w=0.6 |

chosen vs best baseline: MAE +0.030 [-0.008, +0.068], RMSE +0.066 [+0.021, +0.111] *

chosen vs Table 4 LLM: MAE -0.057 [-0.094, -0.020] *, RMSE -0.096 [-0.134, -0.056] *

## Depaul-WH

Table 4: eurollm 1.159 / 1.373; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.100 / 1.338 | 1.157 / 1.371 |  |
| v0+hybrid **chosen** | 0.796 / 1.026 | 0.767 / 0.981 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.028, +0.083] *, RMSE -0.011 [-0.043, +0.021]

chosen vs Table 4 LLM: MAE -0.391 [-0.452, -0.332] *, RMSE -0.392 [-0.463, -0.323] *

## Tijuana-WH

Table 4: eurollm 0.811 / 1.117; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.676 / 1.046 | 0.729 / 1.091 |  |
| v0+hybrid **chosen** | 0.639 / 0.974 | 0.580 / 0.866 | DeepFM w=0.45 |

chosen vs best baseline: MAE +0.082 [+0.040, +0.124] *, RMSE +0.042 [-0.010, +0.094]

chosen vs Table 4 LLM: MAE -0.231 [-0.295, -0.172] *, RMSE -0.251 [-0.334, -0.172] *

## LDOS-H

Table 4: eurollm 0.827 / 1.071; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.762 / 1.025 | 0.823 / 1.058 |  |
| v0+hybrid **chosen** | 0.768 / 1.001 | 0.768 / 0.976 | CAMF_CU w=0.7 |

chosen vs best baseline: MAE +0.023 [-0.015, +0.059], RMSE +0.028 [-0.010, +0.066]

chosen vs Table 4 LLM: MAE -0.059 [-0.096, -0.023] *, RMSE -0.095 [-0.135, -0.056] *

## Depaul-H

Table 4: eurollm 0.735 / 1.051; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.789 / 1.218 | 0.825 / 1.197 |  |
| v0+hybrid **chosen** | 0.760 / 1.021 | 0.749 / 0.983 | NFM w=0.2 |

chosen vs best baseline: MAE +0.036 [+0.010, +0.064] *, RMSE -0.009 [-0.041, +0.024]

chosen vs Table 4 LLM: MAE +0.014 [-0.031, +0.057], RMSE -0.068 [-0.126, -0.012] *

## Restaurant-H

Table 4: eurollm 0.612 / 1.015; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.747 / 1.204 | 0.627 / 1.041 |  |
| v0+hybrid **chosen** | 0.630 / 1.030 | 0.505 / 0.820 | DeepFM w=0.1 |

chosen vs best baseline: MAE +0.008 [-0.001, +0.017], RMSE -0.004 [-0.014, +0.007]

chosen vs Table 4 LLM: MAE -0.107 [-0.182, -0.035] *, RMSE -0.195 [-0.289, -0.103] *

