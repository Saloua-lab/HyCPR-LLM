# Improvement experiments, eurollm (selection on validation, numbers on test)

## LDOS-WH

Table 4: eurollm 0.832 / 1.109; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.794 / 1.077 | 0.825 / 1.101 |  |
| v0+hybrid **chosen** | 0.789 / 1.019 | 0.762 / 0.988 | CAMF_CUCI w=0.55 |

chosen vs best baseline: MAE +0.017 [-0.019, +0.052], RMSE +0.040 [-0.003, +0.086]

chosen vs Table 4 LLM: MAE -0.070 [-0.111, -0.029] *, RMSE -0.121 [-0.163, -0.080] *

## Depaul-WH

Table 4: eurollm 1.159 / 1.373; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.112 / 1.352 | 1.160 / 1.378 |  |
| v0+hybrid **chosen** | 0.797 / 1.028 | 0.768 / 0.982 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.028, +0.083] *, RMSE -0.010 [-0.042, +0.022]

chosen vs Table 4 LLM: MAE -0.391 [-0.451, -0.332] *, RMSE -0.391 [-0.462, -0.322] *

## Tijuana-WH

Table 4: eurollm 0.811 / 1.117; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.721 / 1.100 | 0.729 / 1.107 |  |
| v0+hybrid **chosen** | 0.653 / 1.007 | 0.558 / 0.849 | DeepFM w=0.35 |

chosen vs best baseline: MAE +0.060 [+0.028, +0.093] *, RMSE +0.025 [-0.011, +0.063]

chosen vs Table 4 LLM: MAE -0.253 [-0.320, -0.189] *, RMSE -0.267 [-0.355, -0.183] *

## LDOS-H

Table 4: eurollm 0.827 / 1.071; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.841 / 1.076 | 0.844 / 1.093 |  |
| v0+hybrid **chosen** | 0.833 / 1.043 | 0.759 / 0.967 | CAMF_CUCI w=0.55 |

chosen vs best baseline: MAE +0.014 [-0.021, +0.048], RMSE +0.019 [-0.021, +0.059]

chosen vs Table 4 LLM: MAE -0.068 [-0.106, -0.029] *, RMSE -0.104 [-0.146, -0.064] *

## Depaul-H

Table 4: eurollm 0.735 / 1.051; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.856 / 1.250 | 0.806 / 1.176 |  |
| v0+hybrid **chosen** | 0.773 / 1.021 | 0.744 / 0.976 | NFM w=0.2 |

chosen vs best baseline: MAE +0.031 [+0.005, +0.058] *, RMSE -0.016 [-0.048, +0.016]

chosen vs Table 4 LLM: MAE +0.009 [-0.035, +0.052], RMSE -0.075 [-0.133, -0.019] *

## Restaurant-H

Table 4: eurollm 0.612 / 1.015; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.711 / 1.151 | 0.610 / 1.031 |  |
| v0+hybrid **chosen** | 0.637 / 1.025 | 0.510 / 0.822 | DeepFM w=0.2 |

chosen vs best baseline: MAE +0.012 [-0.005, +0.030], RMSE -0.002 [-0.022, +0.018]

chosen vs Table 4 LLM: MAE -0.102 [-0.173, -0.035] *, RMSE -0.193 [-0.282, -0.103] *

