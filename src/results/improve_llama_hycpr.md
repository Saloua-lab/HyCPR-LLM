# Improvement experiments, llama (selection on validation, numbers on test)

## LDOS-WH

Table 4: llama 0.832 / 1.142; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.870 / 1.172 | 0.822 / 1.121 |  |
| v0+hybrid **chosen** | 0.852 / 1.055 | 0.763 / 0.986 | CAMF_CI w=0.35 |

chosen vs best baseline: MAE +0.018 [-0.014, +0.050], RMSE +0.038 [-0.000, +0.077]

chosen vs Table 4 LLM: MAE -0.069 [-0.112, -0.028] *, RMSE -0.157 [-0.204, -0.109] *

## Depaul-WH

Table 4: llama 1.159 / 1.381; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.110 / 1.347 | 1.163 / 1.379 |  |
| v0+hybrid **chosen** | 0.797 / 1.028 | 0.767 / 0.982 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.028, +0.083] *, RMSE -0.010 [-0.042, +0.022]

chosen vs Table 4 LLM: MAE -0.391 [-0.453, -0.332] *, RMSE -0.399 [-0.471, -0.328] *

## Tijuana-WH

Table 4: llama 0.846 / 1.214; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.750 / 1.149 | 0.814 / 1.182 |  |
| v0+hybrid **chosen** | 0.648 / 1.020 | 0.557 / 0.841 | DeepFM w=0.25 |

chosen vs best baseline: MAE +0.059 [+0.033, +0.086] *, RMSE +0.017 [-0.011, +0.046]

chosen vs Table 4 LLM: MAE -0.289 [-0.367, -0.214] *, RMSE -0.373 [-0.467, -0.284] *

## LDOS-H

Table 4: llama 0.822 / 1.054; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.790 / 1.042 | 0.849 / 1.059 |  |
| v0+hybrid **chosen** | 0.803 / 1.023 | 0.780 / 0.977 | FM w=0.7 |

chosen vs best baseline: MAE +0.035 [-0.003, +0.074], RMSE +0.029 [-0.012, +0.073]

chosen vs Table 4 LLM: MAE -0.042 [-0.074, -0.011] *, RMSE -0.078 [-0.112, -0.043] *

## Depaul-H

Table 4: llama 0.740 / 1.100; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.761 / 1.135 | 0.754 / 1.093 |  |
| v0+hybrid **chosen** | 0.750 / 1.008 | 0.725 / 0.960 | NFM w=0.3 |

chosen vs best baseline: MAE +0.013 [-0.014, +0.040], RMSE -0.032 [-0.066, +0.003]

chosen vs Table 4 LLM: MAE -0.015 [-0.058, +0.028], RMSE -0.140 [-0.206, -0.078] *

## Restaurant-H

Table 4: llama 0.626 / 1.045; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.732 / 1.167 | 0.685 / 1.064 |  |
| v0+hybrid **chosen** | 0.641 / 1.024 | 0.522 / 0.822 | DeepFM w=0.2 |

chosen vs best baseline: MAE +0.025 [+0.006, +0.043] *, RMSE -0.002 [-0.023, +0.018]

chosen vs Table 4 LLM: MAE -0.104 [-0.175, -0.038] *, RMSE -0.224 [-0.316, -0.136] *

