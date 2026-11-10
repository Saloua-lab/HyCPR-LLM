# Improvement experiments, llama (selection on validation, numbers on test)

## LDOS-WH

Table 4: llama 0.832 / 1.142; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.872 / 1.170 | 0.805 / 1.105 |  |
| v0+hybrid **chosen** | 0.847 / 1.059 | 0.757 / 0.982 | CAMF_CI w=0.35 |

chosen vs best baseline: MAE +0.012 [-0.019, +0.044], RMSE +0.035 [-0.004, +0.075]

chosen vs Table 4 LLM: MAE -0.074 [-0.117, -0.034] *, RMSE -0.160 [-0.208, -0.112] *

## Depaul-WH

Table 4: llama 1.159 / 1.381; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.104 / 1.343 | 1.157 / 1.375 |  |
| v0+hybrid **chosen** | 0.796 / 1.027 | 0.767 / 0.981 | NFM w=0.1 |

chosen vs best baseline: MAE +0.055 [+0.027, +0.083] *, RMSE -0.011 [-0.043, +0.021]

chosen vs Table 4 LLM: MAE -0.392 [-0.453, -0.332] *, RMSE -0.399 [-0.472, -0.328] *

## Tijuana-WH

Table 4: llama 0.846 / 1.214; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.774 / 1.149 | 0.748 / 1.112 |  |
| v0+hybrid **chosen** | 0.657 / 1.006 | 0.549 / 0.842 | DeepFM w=0.3 |

chosen vs best baseline: MAE +0.052 [+0.023, +0.080] *, RMSE +0.018 [-0.013, +0.049]

chosen vs Table 4 LLM: MAE -0.297 [-0.374, -0.223] *, RMSE -0.372 [-0.469, -0.282] *

## LDOS-H

Table 4: llama 0.822 / 1.054; best baseline DeepFM 0.745 / 0.947

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.814 / 1.049 | 0.816 / 1.019 |  |
| v0+hybrid **chosen** | 0.826 / 1.033 | 0.768 / 0.954 | CAMF_CI w=0.65 |

chosen vs best baseline: MAE +0.023 [-0.013, +0.059], RMSE +0.007 [-0.034, +0.048]

chosen vs Table 4 LLM: MAE -0.054 [-0.089, -0.020] *, RMSE -0.100 [-0.138, -0.062] *

## Depaul-H

Table 4: llama 0.740 / 1.100; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.747 / 1.085 | 0.799 / 1.160 |  |
| v0+hybrid **chosen** | 0.742 / 1.003 | 0.749 / 0.989 | NFM w=0.4 |

chosen vs best baseline: MAE +0.037 [+0.005, +0.067] *, RMSE -0.003 [-0.043, +0.035]

chosen vs Table 4 LLM: MAE +0.009 [-0.032, +0.049], RMSE -0.112 [-0.175, -0.053] *

## Restaurant-H

Table 4: llama 0.626 / 1.045; best baseline DeepFM 0.498 / 0.824

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.833 / 1.248 | 0.733 / 1.162 |  |
| v0+hybrid **chosen** | 0.650 / 1.026 | 0.522 / 0.833 | DeepFM w=0.15 |

chosen vs best baseline: MAE +0.024 [+0.009, +0.039] *, RMSE +0.009 [-0.007, +0.025]

chosen vs Table 4 LLM: MAE -0.104 [-0.177, -0.036] *, RMSE -0.212 [-0.305, -0.125] *

