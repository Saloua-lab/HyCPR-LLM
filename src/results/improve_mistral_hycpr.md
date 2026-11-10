# Improvement experiments, mistral (selection on validation, numbers on test)

## LDOS-WH

Table 4: mistral 0.764 / 0.954; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.818 / 1.139 | 0.815 / 1.136 |  |
| v0+hybrid **chosen** | 0.831 / 1.056 | 0.756 / 0.986 | CAMF_CU w=0.45 |

chosen vs best baseline: MAE +0.011 [-0.025, +0.048], RMSE +0.039 [-0.000, +0.079]

chosen vs Table 4 LLM: MAE -0.008 [-0.061, +0.045], RMSE +0.032 [-0.025, +0.091]

## Depaul-WH

Table 4: mistral 1.243 / 1.433; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.103 / 1.349 | 1.162 / 1.378 |  |
| v0+hybrid **chosen** | 0.796 / 1.027 | 0.767 / 0.982 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.027, +0.083] *, RMSE -0.010 [-0.042, +0.022]

chosen vs Table 4 LLM: MAE -0.476 [-0.540, -0.413] *, RMSE -0.451 [-0.521, -0.381] *

## Tijuana-WH

Table 4: mistral 0.745 / 1.193; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.729 / 1.161 | 0.709 / 1.103 |  |
| v0+hybrid **chosen** | 0.642 / 1.019 | 0.533 / 0.822 | DeepFM w=0.25 |

chosen vs best baseline: MAE +0.036 [+0.011, +0.062] *, RMSE -0.003 [-0.031, +0.026]

chosen vs Table 4 LLM: MAE -0.212 [-0.301, -0.126] *, RMSE -0.372 [-0.491, -0.256] *

## LDOS-H

Table 4: mistral 0.842 / 1.172; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.819 / 1.060 | 0.764 / 1.010 |  |
| v0+hybrid **chosen** | 0.824 / 1.034 | 0.711 / 0.920 | CAMF_CUCI w=0.6 |

chosen vs best baseline: MAE -0.034 [-0.070, +0.003], RMSE -0.027 [-0.068, +0.015]

chosen vs Table 4 LLM: MAE -0.131 [-0.188, -0.076] *, RMSE -0.252 [-0.314, -0.188] *

## Depaul-H

Table 4: mistral 0.795 / 1.190; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.669 / 1.039 | 0.763 / 1.153 |  |
| v0+hybrid **chosen** | 0.681 / 0.977 | 0.711 / 0.999 | CAMF_CU w=0.55 |

chosen vs best baseline: MAE -0.001 [-0.033, +0.032], RMSE +0.007 [-0.033, +0.048]

chosen vs Table 4 LLM: MAE -0.083 [-0.131, -0.037] *, RMSE -0.191 [-0.253, -0.129] *

## Restaurant-H

Table 4: mistral 0.598 / 1.036; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.623 / 1.076 | 0.588 / 1.002 |  |
| v0+hybrid **chosen** | 0.619 / 0.997 | 0.523 / 0.840 | DeepFM w=0.4 |

chosen vs best baseline: MAE +0.025 [-0.006, +0.057], RMSE +0.016 [-0.022, +0.057]

chosen vs Table 4 LLM: MAE -0.075 [-0.146, -0.006] *, RMSE -0.195 [-0.306, -0.094] *

