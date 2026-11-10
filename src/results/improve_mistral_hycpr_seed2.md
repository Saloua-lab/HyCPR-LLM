# Improvement experiments, mistral (selection on validation, numbers on test)

## LDOS-WH

Table 4: mistral 0.764 / 0.954; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.835 / 1.154 | 0.790 / 1.106 |  |
| v0+hybrid **chosen** | 0.855 / 1.068 | 0.760 / 0.985 | CAMF_CI w=0.3 |

chosen vs best baseline: MAE +0.015 [-0.016, +0.046], RMSE +0.037 [+0.001, +0.074] *

chosen vs Table 4 LLM: MAE -0.004 [-0.052, +0.044], RMSE +0.030 [-0.023, +0.085]

## Depaul-WH

Table 4: mistral 1.243 / 1.433; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.110 / 1.348 | 1.165 / 1.381 |  |
| v0+hybrid **chosen** | 0.796 / 1.027 | 0.767 / 0.982 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.028, +0.083] *, RMSE -0.010 [-0.042, +0.022]

chosen vs Table 4 LLM: MAE -0.476 [-0.539, -0.413] *, RMSE -0.451 [-0.521, -0.381] *

## Tijuana-WH

Table 4: mistral 0.745 / 1.193; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.694 / 1.106 | 0.652 / 1.049 |  |
| v0+hybrid **chosen** | 0.640 / 1.017 | 0.528 / 0.828 | DeepFM w=0.3 |

chosen vs best baseline: MAE +0.031 [+0.004, +0.059] *, RMSE +0.004 [-0.025, +0.032]

chosen vs Table 4 LLM: MAE -0.217 [-0.311, -0.128] *, RMSE -0.365 [-0.491, -0.246] *

## LDOS-H

Table 4: mistral 0.842 / 1.172; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.758 / 1.082 | 0.808 / 1.128 |  |
| v0+hybrid **chosen** | 0.799 / 1.036 | 0.756 / 1.006 | FM w=0.55 |

chosen vs best baseline: MAE +0.011 [-0.029, +0.051], RMSE +0.058 [+0.011, +0.105] *

chosen vs Table 4 LLM: MAE -0.086 [-0.132, -0.040] *, RMSE -0.167 [-0.219, -0.115] *

## Depaul-H

Table 4: mistral 0.795 / 1.190; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.715 / 1.168 | 0.781 / 1.173 |  |
| v0+hybrid **chosen** | 0.735 / 1.006 | 0.736 / 0.974 | NFM w=0.3 |

chosen vs best baseline: MAE +0.024 [-0.006, +0.055], RMSE -0.018 [-0.055, +0.019]

chosen vs Table 4 LLM: MAE -0.059 [-0.107, -0.010] *, RMSE -0.215 [-0.279, -0.151] *

## Restaurant-H

Table 4: mistral 0.598 / 1.036; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.518 / 0.936 | 0.584 / 1.067 |  |
| v0+hybrid **chosen** | 0.545 / 0.907 | 0.546 / 0.945 | DeepFM w=0.7 |

chosen vs best baseline: MAE +0.048 [-0.011, +0.108], RMSE +0.121 [+0.036, +0.203] *

chosen vs Table 4 LLM: MAE -0.052 [-0.117, +0.012], RMSE -0.091 [-0.189, +0.004]

