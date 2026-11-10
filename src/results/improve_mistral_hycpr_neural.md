# Improvement experiments, mistral (selection on validation, numbers on test)

## LDOS-WH

Table 4: mistral 0.764 / 0.954; best baseline CAMF_C 0.757 / 0.956

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.818 / 1.139 | 0.815 / 1.136 |  |
| v0+hybrid **chosen** | 0.831 / 1.056 | 0.759 / 0.988 | CAMF_CU w=0.45 |

chosen vs best baseline: MAE +0.002 [-0.036, +0.040], RMSE +0.031 [-0.012, +0.074]

chosen vs Table 4 LLM: MAE -0.004 [-0.056, +0.048], RMSE +0.033 [-0.024, +0.091]

## Depaul-WH

Table 4: mistral 1.243 / 1.433; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.103 / 1.349 | 1.162 / 1.378 |  |
| v0+hybrid **chosen** | 0.796 / 1.027 | 0.814 / 1.029 | NFM w=0.1 |

chosen vs best baseline: MAE +0.101 [+0.066, +0.139] *, RMSE +0.037 [-0.011, +0.087]

chosen vs Table 4 LLM: MAE -0.429 [-0.490, -0.370] *, RMSE -0.404 [-0.469, -0.338] *

## Tijuana-WH

Table 4: mistral 0.745 / 1.193; best baseline FM 0.748 / 1.073

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.729 / 1.161 | 0.709 / 1.103 |  |
| v0+hybrid **chosen** | 0.642 / 1.019 | 0.589 / 0.904 | DeepFM w=0.25 |

chosen vs best baseline: MAE -0.158 [-0.226, -0.091] *, RMSE -0.170 [-0.262, -0.082] *

chosen vs Table 4 LLM: MAE -0.156 [-0.244, -0.072] *, RMSE -0.290 [-0.409, -0.174] *

## LDOS-H

Table 4: mistral 0.842 / 1.172; best baseline CAMF_C 0.757 / 0.956

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.819 / 1.060 | 0.764 / 1.010 |  |
| v0+hybrid **chosen** | 0.824 / 1.034 | 0.708 / 0.918 | CAMF_CUCI w=0.6 |

chosen vs best baseline: MAE -0.049 [-0.089, -0.009] *, RMSE -0.038 [-0.084, +0.007]

chosen vs Table 4 LLM: MAE -0.134 [-0.191, -0.078] *, RMSE -0.254 [-0.318, -0.190] *

## Depaul-H

Table 4: mistral 0.795 / 1.190; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.669 / 1.039 | 0.763 / 1.153 |  |
| v0+hybrid **chosen** | 0.681 / 0.977 | 0.726 / 1.031 | CAMF_CU w=0.55 |

chosen vs best baseline: MAE +0.014 [-0.024, +0.053], RMSE +0.039 [-0.013, +0.095]

chosen vs Table 4 LLM: MAE -0.069 [-0.118, -0.020] *, RMSE -0.158 [-0.223, -0.093] *

## Restaurant-H

Table 4: mistral 0.598 / 1.036; best baseline FM 0.748 / 1.073

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.623 / 1.076 | 0.588 / 1.002 |  |
| v0+hybrid **chosen** | 0.619 / 0.997 | 0.568 / 0.900 | DeepFM w=0.4 |

chosen vs best baseline: MAE -0.180 [-0.248, -0.112] *, RMSE -0.173 [-0.272, -0.078] *

chosen vs Table 4 LLM: MAE -0.030 [-0.101, +0.039], RMSE -0.136 [-0.241, -0.037] *

