# Improvement experiments, mistral (selection on validation, numbers on test)

## LDOS-WH

Table 4: mistral 0.764 / 0.954; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.821 / 1.151 | 0.800 / 1.119 |  |
| v0+hybrid **chosen** | 0.828 / 1.059 | 0.753 / 0.977 | CAMF_CU w=0.4 |

chosen vs best baseline: MAE +0.008 [-0.028, +0.044], RMSE +0.029 [-0.011, +0.071]

chosen vs Table 4 LLM: MAE -0.011 [-0.063, +0.042], RMSE +0.022 [-0.036, +0.083]

## Depaul-WH

Table 4: mistral 1.243 / 1.433; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.107 / 1.348 | 1.162 / 1.380 |  |
| v0+hybrid **chosen** | 0.796 / 1.027 | 0.767 / 0.982 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.028, +0.083] *, RMSE -0.010 [-0.042, +0.022]

chosen vs Table 4 LLM: MAE -0.476 [-0.539, -0.413] *, RMSE -0.451 [-0.521, -0.382] *

## Tijuana-WH

Table 4: mistral 0.745 / 1.193; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.704 / 1.074 | 0.729 / 1.095 |  |
| v0+hybrid **chosen** | 0.647 / 1.001 | 0.568 / 0.856 | DeepFM w=0.4 |

chosen vs best baseline: MAE +0.070 [+0.033, +0.108] *, RMSE +0.032 [-0.011, +0.077]

chosen vs Table 4 LLM: MAE -0.177 [-0.267, -0.093] *, RMSE -0.337 [-0.454, -0.226] *

## LDOS-H

Table 4: mistral 0.842 / 1.172; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.787 / 1.100 | 0.804 / 1.103 |  |
| v0+hybrid **chosen** | 0.831 / 1.037 | 0.753 / 0.969 | GAR w=0.55 |

chosen vs best baseline: MAE +0.008 [-0.031, +0.047], RMSE +0.021 [-0.022, +0.064]

chosen vs Table 4 LLM: MAE -0.089 [-0.146, -0.034] *, RMSE -0.204 [-0.266, -0.141] *

## Depaul-H

Table 4: mistral 0.795 / 1.190; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.728 / 1.136 | 0.762 / 1.134 |  |
| v0+hybrid **chosen** | 0.732 / 0.997 | 0.728 / 0.967 | NFM w=0.35 |

chosen vs best baseline: MAE +0.016 [-0.013, +0.046], RMSE -0.025 [-0.064, +0.016]

chosen vs Table 4 LLM: MAE -0.067 [-0.117, -0.017] *, RMSE -0.222 [-0.289, -0.157] *

## Restaurant-H

Table 4: mistral 0.598 / 1.036; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.685 / 1.131 | 0.643 / 1.084 |  |
| v0+hybrid **chosen** | 0.636 / 1.001 | 0.537 / 0.857 | DeepFM w=0.35 |

chosen vs best baseline: MAE +0.040 [+0.009, +0.071] *, RMSE +0.033 [-0.009, +0.074]

chosen vs Table 4 LLM: MAE -0.060 [-0.129, +0.007], RMSE -0.179 [-0.274, -0.086] *

