# Improvement experiments, llama (selection on validation, numbers on test)

## LDOS-WH

Table 4: llama 0.832 / 1.142; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.846 / 1.160 | 0.801 / 1.091 |  |
| v0+hybrid **chosen** | 0.846 / 1.056 | 0.759 / 0.980 | CAMF_CI w=0.35 |

chosen vs best baseline: MAE +0.014 [-0.017, +0.045], RMSE +0.033 [-0.002, +0.069]

chosen vs Table 4 LLM: MAE -0.073 [-0.117, -0.031] *, RMSE -0.162 [-0.212, -0.112] *

## Depaul-WH

Table 4: llama 1.159 / 1.381; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.107 / 1.346 | 1.159 / 1.375 |  |
| v0+hybrid **chosen** | 0.796 / 1.027 | 0.767 / 0.981 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.027, +0.082] *, RMSE -0.011 [-0.043, +0.021]

chosen vs Table 4 LLM: MAE -0.392 [-0.453, -0.332] *, RMSE -0.400 [-0.472, -0.328] *

## Tijuana-WH

Table 4: llama 0.846 / 1.214; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.849 / 1.229 | 0.889 / 1.225 |  |
| v0+hybrid **chosen** | 0.641 / 1.029 | 0.528 / 0.829 | DeepFM w=0.1 |

chosen vs best baseline: MAE +0.030 [+0.019, +0.041] *, RMSE +0.005 [-0.005, +0.015]

chosen vs Table 4 LLM: MAE -0.318 [-0.407, -0.234] *, RMSE -0.386 [-0.491, -0.288] *

## LDOS-H

Table 4: llama 0.822 / 1.054; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.824 / 1.049 | 0.807 / 1.000 |  |
| v0+hybrid **chosen** | 0.829 / 1.038 | 0.755 / 0.941 | FM w=0.7 |

chosen vs best baseline: MAE +0.010 [-0.024, +0.044], RMSE -0.006 [-0.043, +0.033]

chosen vs Table 4 LLM: MAE -0.067 [-0.104, -0.031] *, RMSE -0.113 [-0.151, -0.075] *

## Depaul-H

Table 4: llama 0.740 / 1.100; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.755 / 1.097 | 0.777 / 1.119 |  |
| v0+hybrid **chosen** | 0.734 / 1.005 | 0.721 / 0.965 | FM w=0.45 |

chosen vs best baseline: MAE +0.009 [-0.025, +0.043], RMSE -0.027 [-0.070, +0.015]

chosen vs Table 4 LLM: MAE -0.019 [-0.065, +0.025], RMSE -0.136 [-0.203, -0.074] *

## Restaurant-H

Table 4: llama 0.626 / 1.045; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.753 / 1.209 | 0.688 / 1.111 |  |
| v0+hybrid **chosen** | 0.637 / 1.025 | 0.518 / 0.828 | DeepFM w=0.15 |

chosen vs best baseline: MAE +0.020 [+0.006, +0.035] *, RMSE +0.004 [-0.011, +0.020]

chosen vs Table 4 LLM: MAE -0.108 [-0.181, -0.041] *, RMSE -0.217 [-0.310, -0.130] *

