# Improvement experiments, mistral (selection on validation, numbers on test)

## LDOS-WH

Table 4: mistral 0.764 / 0.954; best baseline CAMF_C 0.757 / 0.956

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.818 / 1.139 | 0.815 / 1.136 |  |
| v0+hybrid **chosen** | 0.831 / 1.056 | 0.759 / 0.988 | CAMF_CU w=0.45 |
| vr | 0.900 / 1.131 | 0.775 / 0.998 |  |
| vr+hybrid | 0.872 / 1.075 | 0.756 / 0.933 | FM w=0.45 |
| vrcf | 0.930 / 1.134 | 0.796 / 0.996 |  |
| vrcf+hybrid | 0.882 / 1.078 | 0.759 / 0.934 | FM w=0.45 |
| v0ens | 0.821 / 1.132 | 0.793 / 1.103 |  |
| v0ens+hybrid | 0.828 / 1.058 | 0.753 / 0.977 | CAMF_CU w=0.45 |

chosen vs best baseline: MAE +0.002 [-0.036, +0.040], RMSE +0.031 [-0.012, +0.074]

chosen vs Table 4 LLM: MAE -0.004 [-0.056, +0.048], RMSE +0.033 [-0.024, +0.091]

## Depaul-WH

Table 4: mistral 1.243 / 1.433; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 1.103 / 1.349 | 1.162 / 1.378 |  |
| v0+hybrid | 0.783 / 1.031 | 0.795 / 1.037 | CAMF_CU w=0.2 |
| vr | 1.270 / 1.463 | 1.286 / 1.478 |  |
| vr+hybrid | 0.806 / 1.036 | 0.810 / 1.041 | CAMF_CU w=0.2 |
| vrcf | 0.885 / 1.113 | 1.000 / 1.238 |  |
| vrcf+hybrid **chosen** | 0.773 / 1.026 | 0.794 / 1.047 | CAMF_CU w=0.4 |
| v0ens | 1.106 / 1.347 | 1.162 / 1.379 |  |
| v0ens+hybrid | 0.782 / 1.031 | 0.795 / 1.037 | CAMF_CU w=0.2 |

chosen vs best baseline: MAE +0.082 [+0.046, +0.118] *, RMSE +0.055 [+0.014, +0.099] *

chosen vs Table 4 LLM: MAE -0.449 [-0.514, -0.384] *, RMSE -0.385 [-0.460, -0.313] *

## Tijuana-WH

Table 4: mistral 0.745 / 1.193; best baseline FM 0.748 / 1.073

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.729 / 1.161 | 0.709 / 1.103 |  |
| v0+hybrid | 0.740 / 1.126 | 0.724 / 1.039 | CAMF_CU w=0.55 |
| vr | 1.155 / 1.455 | 1.062 / 1.353 |  |
| vr+hybrid | 0.844 / 1.147 | 0.819 / 1.068 | CAMF_CU w=0.2 |
| vrcf | 0.894 / 1.249 | 0.885 / 1.231 |  |
| vrcf+hybrid | 0.812 / 1.131 | 0.781 / 1.062 | CAMF_CU w=0.35 |
| v0ens | 0.704 / 1.077 | 0.690 / 1.037 |  |
| v0ens+hybrid **chosen** | 0.712 / 1.074 | 0.698 / 1.021 | CAMF_CU w=0.85 |

chosen vs best baseline: MAE -0.050 [-0.136, +0.037], RMSE -0.052 [-0.162, +0.059]

chosen vs Table 4 LLM: MAE -0.047 [-0.118, +0.020], RMSE -0.172 [-0.261, -0.086] *

## LDOS-H

Table 4: mistral 0.842 / 1.172; best baseline CAMF_C 0.757 / 0.956

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.819 / 1.060 | 0.764 / 1.010 |  |
| v0+hybrid | 0.824 / 1.034 | 0.708 / 0.918 | CAMF_CUCI w=0.6 |
| vr | 0.912 / 1.132 | 0.784 / 0.998 |  |
| vr+hybrid | 0.876 / 1.076 | 0.757 / 0.934 | FM w=0.45 |
| vrcf | 0.897 / 1.109 | 0.766 / 0.971 |  |
| vrcf+hybrid | 0.871 / 1.075 | 0.754 / 0.932 | FM w=0.5 |
| v0ens | 0.768 / 1.023 | 0.756 / 1.020 |  |
| v0ens+hybrid **chosen** | 0.779 / 1.012 | 0.728 / 0.968 | FM w=0.75 |

chosen vs best baseline: MAE -0.029 [-0.069, +0.012], RMSE +0.012 [-0.035, +0.059]

chosen vs Table 4 LLM: MAE -0.114 [-0.161, -0.069] *, RMSE -0.204 [-0.256, -0.153] *

## Depaul-H

Table 4: mistral 0.795 / 1.190; best baseline CAMF_CU 0.712 / 0.992

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.669 / 1.039 | 0.763 / 1.153 |  |
| v0+hybrid **chosen** | 0.681 / 0.977 | 0.726 / 1.031 | CAMF_CU w=0.55 |
| vr | 1.257 / 1.457 | 1.279 / 1.481 |  |
| vr+hybrid | 0.805 / 1.035 | 0.809 / 1.041 | CAMF_CU w=0.2 |
| vrcf | 1.267 / 1.461 | 1.286 / 1.478 |  |
| vrcf+hybrid | 0.806 / 1.036 | 0.810 / 1.041 | CAMF_CU w=0.2 |
| v0ens | 0.678 / 1.036 | 0.736 / 1.071 |  |
| v0ens+hybrid | 0.689 / 0.983 | 0.716 / 1.008 | CAMF_CU w=0.55 |

chosen vs best baseline: MAE +0.014 [-0.024, +0.053], RMSE +0.039 [-0.013, +0.095]

chosen vs Table 4 LLM: MAE -0.069 [-0.118, -0.020] *, RMSE -0.158 [-0.223, -0.093] *

## Restaurant-H

Table 4: mistral 0.598 / 1.036; best baseline FM 0.748 / 1.073

| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |
|---|---|---|---|
| v0 | 0.623 / 1.076 | 0.588 / 1.002 |  |
| v0+hybrid | 0.723 / 1.057 | 0.680 / 0.991 | GAR w=0.85 |
| vr | 0.903 / 1.298 | 0.858 / 1.226 |  |
| vr+hybrid | 0.811 / 1.139 | 0.784 / 1.057 | CAMF_CU w=0.3 |
| vrcf | 0.875 / 1.223 | 0.821 / 1.166 |  |
| vrcf+hybrid | 0.802 / 1.129 | 0.765 / 1.044 | CAMF_CU w=0.4 |
| v0ens | 0.600 / 1.006 | 0.594 / 1.005 |  |
| v0ens+hybrid **chosen** | 0.632 / 1.002 | 0.624 / 0.997 | GAR w=0.95 |

chosen vs best baseline: MAE -0.124 [-0.208, -0.042] *, RMSE -0.077 [-0.192, +0.040]

chosen vs Table 4 LLM: MAE +0.026 [-0.032, +0.084], RMSE -0.039 [-0.132, +0.049]

