# Improvement experiments, eurollm (selection on validation, numbers on test)

## LDOS-WH

Table 4: eurollm 0.832 / 1.109; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.789 / 1.056 | 0.787 / 1.085 |  |
| v0+hybrid **chosen** | 0.777 / 1.008 | 0.751 / 0.980 | CAMF_CU w=0.6 |

chosen vs best baseline: MAE +0.006 [-0.032, +0.044], RMSE +0.033 [-0.012, +0.079]

chosen vs Table 4 LLM: MAE -0.081 [-0.123, -0.040] *, RMSE -0.129 [-0.170, -0.087] *

## Depaul-WH

Table 4: eurollm 1.159 / 1.373; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.108 / 1.342 | 1.162 / 1.380 |  |
| v0+hybrid **chosen** | 0.796 / 1.027 | 0.768 / 0.981 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.028, +0.083] *, RMSE -0.011 [-0.043, +0.022]

chosen vs Table 4 LLM: MAE -0.391 [-0.451, -0.332] *, RMSE -0.392 [-0.462, -0.322] *

## Tijuana-WH

Table 4: eurollm 0.811 / 1.117; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.684 / 1.103 | 0.718 / 1.115 |  |
| v0+hybrid **chosen** | 0.637 / 0.987 | 0.566 / 0.860 | DeepFM w=0.4 |

chosen vs best baseline: MAE +0.068 [+0.030, +0.108] *, RMSE +0.035 [-0.010, +0.082]

chosen vs Table 4 LLM: MAE -0.245 [-0.311, -0.182] *, RMSE -0.257 [-0.343, -0.175] *

## LDOS-H

Table 4: eurollm 0.827 / 1.071; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.798 / 1.034 | 0.827 / 1.066 |  |
| v0+hybrid **chosen** | 0.793 / 1.017 | 0.762 / 0.967 | CAMF_CU w=0.7 |

chosen vs best baseline: MAE +0.017 [-0.022, +0.057], RMSE +0.020 [-0.023, +0.063]

chosen vs Table 4 LLM: MAE -0.064 [-0.099, -0.030] *, RMSE -0.104 [-0.138, -0.068] *

## Depaul-H

Table 4: eurollm 0.735 / 1.051; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.774 / 1.170 | 0.756 / 1.107 |  |
| v0+hybrid **chosen** | 0.742 / 1.007 | 0.729 / 0.966 | NFM w=0.3 |

chosen vs best baseline: MAE +0.017 [-0.012, +0.045], RMSE -0.026 [-0.063, +0.010]

chosen vs Table 4 LLM: MAE -0.006 [-0.049, +0.035], RMSE -0.085 [-0.142, -0.032] *

## Restaurant-H

Table 4: eurollm 0.612 / 1.015; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.715 / 1.142 | 0.699 / 1.123 |  |
| v0+hybrid **chosen** | 0.642 / 1.020 | 0.533 / 0.840 | DeepFM w=0.25 |

chosen vs best baseline: MAE +0.035 [+0.013, +0.059] *, RMSE +0.016 [-0.011, +0.043]

chosen vs Table 4 LLM: MAE -0.079 [-0.145, -0.016] *, RMSE -0.175 [-0.258, -0.093] *

