# Bootstrap over test interactions (B=10000, seed 0)

## LDOS-WH (n=452)

| model | MAE [95% CI] | RMSE [95% CI] |
|---|---|---|
| DeepFM | 0.745 [0.691, 0.800] | 0.947 [0.881, 1.014] |
| CAMF_C | 0.757 [0.703, 0.812] | 0.956 [0.887, 1.024] |
| NFM | 0.761 [0.708, 0.816] | 0.960 [0.893, 1.026] |
| BiasedMF | 0.763 [0.709, 0.818] | 0.961 [0.894, 1.028] |
| mistral | 0.764 [0.712, 0.818] | 0.954 [0.889, 1.019] |
| SVD++ | 0.767 [0.711, 0.823] | 0.974 [0.904, 1.043] |
| CAMF_CI | 0.777 [0.722, 0.835] | 0.988 [0.917, 1.058] |
| FM | 0.778 [0.721, 0.836] | 0.999 [0.925, 1.072] |
| AutoInt | 0.781 [0.729, 0.834] | 0.963 [0.903, 1.024] |
| CAMF_CUCI | 0.781 [0.725, 0.840] | 0.998 [0.927, 1.066] |
| UAR | 0.785 [0.735, 0.838] | 0.960 [0.900, 1.020] |
| CAMF_CU | 0.790 [0.731, 0.847] | 1.010 [0.940, 1.078] |
| deepseek | 0.808 [0.740, 0.879] | 1.097 [1.007, 1.188] |
| GAR | 0.809 [0.756, 0.864] | 0.997 [0.936, 1.059] |
| ItemKNN | 0.812 [0.738, 0.887] | 1.133 [1.048, 1.218] |
| IAR | 0.816 [0.743, 0.891] | 1.130 [1.044, 1.216] |
| UserKNN | 0.822 [0.757, 0.886] | 1.082 [1.000, 1.160] |
| eurollm | 0.832 [0.763, 0.901] | 1.109 [1.030, 1.188] |
| llama | 0.832 [0.760, 0.906] | 1.142 [1.055, 1.229] |

## Depaul-WH (n=684)

| model | MAE [95% CI] | RMSE [95% CI] |
|---|---|---|
| CAMF_CU | 0.712 [0.661, 0.765] | 0.992 [0.922, 1.060] |
| CAMF_CUCI | 0.733 [0.682, 0.786] | 1.004 [0.935, 1.071] |
| BiasedMF | 0.734 [0.679, 0.789] | 1.034 [0.959, 1.108] |
| FM | 0.734 [0.685, 0.784] | 0.986 [0.922, 1.049] |
| SVD++ | 0.735 [0.680, 0.791] | 1.036 [0.960, 1.109] |
| CAMF_C | 0.737 [0.682, 0.794] | 1.038 [0.963, 1.112] |
| NFM | 0.744 [0.695, 0.793] | 0.986 [0.921, 1.049] |
| CAMF_CI | 0.760 [0.704, 0.816] | 1.064 [0.989, 1.137] |
| DeepFM | 0.823 [0.772, 0.875] | 1.076 [1.009, 1.140] |
| AutoInt | 0.841 [0.791, 0.893] | 1.084 [1.017, 1.151] |
| ItemKNN | 0.860 [0.810, 0.911] | 1.088 [1.028, 1.149] |
| UserKNN | 0.895 [0.841, 0.948] | 1.145 [1.080, 1.209] |
| UAR | 1.131 [1.077, 1.188] | 1.360 [1.303, 1.418] |
| deepseek | 1.154 [1.098, 1.210] | 1.375 [1.319, 1.429] |
| eurollm | 1.159 [1.103, 1.214] | 1.373 [1.318, 1.427] |
| llama | 1.159 [1.102, 1.215] | 1.381 [1.323, 1.436] |
| IAR | 1.159 [1.103, 1.215] | 1.377 [1.321, 1.431] |
| mistral | 1.243 [1.190, 1.297] | 1.433 [1.385, 1.479] |
| GAR | 1.285 [1.230, 1.340] | 1.478 [1.431, 1.524] |

## Tijuana-WH (n=250)

| model | MAE [95% CI] | RMSE [95% CI] |
|---|---|---|
| DeepFM | 0.498 [0.419, 0.582] | 0.824 [0.698, 0.948] |
| NFM | 0.618 [0.527, 0.715] | 0.976 [0.842, 1.109] |
| mistral | 0.745 [0.630, 0.864] | 1.193 [1.049, 1.336] |
| FM | 0.748 [0.656, 0.845] | 1.073 [0.950, 1.198] |
| CAMF_CU | 0.762 [0.671, 0.856] | 1.072 [0.961, 1.185] |
| deepseek | 0.764 [0.670, 0.865] | 1.097 [0.981, 1.214] |
| CAMF_C | 0.798 [0.703, 0.898] | 1.124 [1.020, 1.228] |
| BiasedMF | 0.810 [0.703, 0.920] | 1.198 [1.074, 1.320] |
| AutoInt | 0.811 [0.712, 0.913] | 1.149 [1.035, 1.267] |
| eurollm | 0.811 [0.719, 0.908] | 1.117 [1.011, 1.220] |
| SVD++ | 0.823 [0.715, 0.935] | 1.213 [1.087, 1.336] |
| UAR | 0.834 [0.731, 0.942] | 1.196 [1.091, 1.301] |
| llama | 0.846 [0.741, 0.957] | 1.214 [1.093, 1.336] |
| CAMF_CUCI | 0.860 [0.766, 0.958] | 1.159 [1.042, 1.278] |
| UserKNN | 0.870 [0.745, 1.001] | 1.343 [1.191, 1.492] |
| CAMF_CI | 0.880 [0.782, 0.984] | 1.203 [1.090, 1.316] |
| ItemKNN | 0.910 [0.802, 1.023] | 1.277 [1.153, 1.402] |
| GAR | 1.234 [1.140, 1.332] | 1.459 [1.351, 1.566] |
| IAR | 1.249 [1.152, 1.351] | 1.490 [1.378, 1.602] |

## LDOS-H (n=452)

| model | MAE [95% CI] | RMSE [95% CI] |
|---|---|---|
| DeepFM | 0.745 [0.691, 0.800] | 0.947 [0.881, 1.014] |
| CAMF_C | 0.757 [0.703, 0.812] | 0.956 [0.887, 1.024] |
| NFM | 0.761 [0.708, 0.816] | 0.960 [0.893, 1.026] |
| BiasedMF | 0.763 [0.709, 0.818] | 0.961 [0.894, 1.028] |
| SVD++ | 0.767 [0.711, 0.823] | 0.974 [0.904, 1.043] |
| CAMF_CI | 0.777 [0.722, 0.835] | 0.988 [0.917, 1.058] |
| FM | 0.778 [0.721, 0.836] | 0.999 [0.925, 1.072] |
| AutoInt | 0.781 [0.729, 0.834] | 0.963 [0.903, 1.024] |
| CAMF_CUCI | 0.781 [0.725, 0.840] | 0.998 [0.927, 1.066] |
| UAR | 0.785 [0.735, 0.838] | 0.960 [0.900, 1.020] |
| CAMF_CU | 0.790 [0.731, 0.847] | 1.010 [0.940, 1.078] |
| deepseek_noitclean | 0.801 [0.740, 0.863] | 1.038 [0.962, 1.115] |
| deepseek_clean | 0.803 [0.744, 0.864] | 1.037 [0.964, 1.109] |
| GAR | 0.809 [0.756, 0.864] | 0.997 [0.936, 1.059] |
| ItemKNN | 0.812 [0.738, 0.887] | 1.133 [1.048, 1.218] |
| IAR | 0.816 [0.743, 0.891] | 1.130 [1.044, 1.216] |
| eurollm | 0.816 [0.754, 0.879] | 1.064 [0.987, 1.138] |
| llama_noitclean | 0.819 [0.756, 0.883] | 1.072 [0.992, 1.150] |
| UserKNN | 0.822 [0.757, 0.886] | 1.082 [1.000, 1.160] |
| llama_clean | 0.822 [0.761, 0.884] | 1.054 [0.979, 1.129] |
| eurollm_clean | 0.827 [0.763, 0.890] | 1.071 [0.994, 1.146] |
| mistral_noitclean | 0.833 [0.759, 0.906] | 1.153 [1.063, 1.241] |
| llama | 0.835 [0.771, 0.900] | 1.081 [1.003, 1.159] |
| mistral_clean | 0.842 [0.767, 0.919] | 1.172 [1.080, 1.266] |
| eurollm_noitclean | 0.860 [0.806, 0.913] | 1.039 [0.977, 1.101] |
| deepseek | 0.864 [0.795, 0.933] | 1.140 [1.057, 1.221] |
| mistral | 0.920 [0.849, 0.993] | 1.213 [1.117, 1.306] |

## Depaul-H (n=684)

| model | MAE [95% CI] | RMSE [95% CI] |
|---|---|---|
| CAMF_CU | 0.712 [0.661, 0.765] | 0.992 [0.922, 1.060] |
| mistral_noitclean | 0.729 [0.665, 0.795] | 1.119 [1.035, 1.202] |
| CAMF_CUCI | 0.733 [0.682, 0.786] | 1.004 [0.935, 1.071] |
| BiasedMF | 0.734 [0.679, 0.789] | 1.034 [0.959, 1.108] |
| FM | 0.734 [0.685, 0.784] | 0.986 [0.922, 1.049] |
| SVD++ | 0.735 [0.680, 0.791] | 1.036 [0.960, 1.109] |
| eurollm_clean | 0.735 [0.678, 0.794] | 1.051 [0.977, 1.125] |
| CAMF_C | 0.737 [0.682, 0.794] | 1.038 [0.963, 1.112] |
| llama_clean | 0.740 [0.679, 0.803] | 1.100 [1.018, 1.182] |
| NFM | 0.744 [0.695, 0.793] | 0.986 [0.921, 1.049] |
| llama_noitclean | 0.745 [0.684, 0.808] | 1.114 [1.033, 1.195] |
| CAMF_CI | 0.760 [0.704, 0.816] | 1.064 [0.989, 1.137] |
| llama | 0.783 [0.721, 0.844] | 1.127 [1.040, 1.210] |
| mistral_clean | 0.795 [0.729, 0.862] | 1.190 [1.103, 1.274] |
| deepseek_noitclean | 0.798 [0.742, 0.858] | 1.110 [1.036, 1.187] |
| deepseek_clean | 0.799 [0.741, 0.856] | 1.118 [1.045, 1.189] |
| eurollm_noitclean | 0.805 [0.744, 0.867] | 1.145 [1.064, 1.222] |
| deepseek | 0.815 [0.755, 0.877] | 1.142 [1.067, 1.217] |
| mistral | 0.821 [0.759, 0.884] | 1.169 [1.091, 1.246] |
| DeepFM | 0.823 [0.772, 0.875] | 1.076 [1.009, 1.140] |
| AutoInt | 0.841 [0.791, 0.893] | 1.084 [1.017, 1.151] |
| ItemKNN | 0.860 [0.810, 0.911] | 1.088 [1.028, 1.149] |
| eurollm | 0.866 [0.802, 0.932] | 1.216 [1.137, 1.295] |
| UserKNN | 0.895 [0.841, 0.948] | 1.145 [1.080, 1.209] |
| UAR | 1.131 [1.077, 1.188] | 1.360 [1.303, 1.418] |
| IAR | 1.159 [1.103, 1.215] | 1.377 [1.321, 1.431] |
| GAR | 1.285 [1.230, 1.340] | 1.478 [1.431, 1.524] |

## Restaurant-H (n=250)

| model | MAE [95% CI] | RMSE [95% CI] |
|---|---|---|
| DeepFM | 0.498 [0.419, 0.582] | 0.824 [0.698, 0.948] |
| mistral | 0.574 [0.483, 0.670] | 0.952 [0.824, 1.075] |
| mistral_clean | 0.598 [0.496, 0.706] | 1.036 [0.899, 1.170] |
| eurollm_clean | 0.612 [0.514, 0.715] | 1.015 [0.881, 1.147] |
| NFM | 0.618 [0.527, 0.715] | 0.976 [0.842, 1.109] |
| mistral_noitclean | 0.621 [0.515, 0.731] | 1.074 [0.931, 1.212] |
| llama_clean | 0.626 [0.526, 0.733] | 1.045 [0.913, 1.176] |
| deepseek | 0.638 [0.543, 0.740] | 1.021 [0.900, 1.142] |
| mistral_noit | 0.639 [0.541, 0.743] | 1.044 [0.909, 1.174] |
| eurollm_noitclean | 0.644 [0.541, 0.752] | 1.067 [0.937, 1.193] |
| llama_noitclean | 0.653 [0.558, 0.755] | 1.027 [0.912, 1.144] |
| eurollm | 0.659 [0.567, 0.757] | 1.021 [0.899, 1.143] |
| deepseek_clean | 0.675 [0.570, 0.789] | 1.108 [0.975, 1.240] |
| llama | 0.704 [0.599, 0.814] | 1.118 [0.992, 1.243] |
| deepseek_noitclean | 0.709 [0.606, 0.818] | 1.109 [0.985, 1.233] |
| FM | 0.748 [0.656, 0.845] | 1.073 [0.950, 1.198] |
| CAMF_CU | 0.762 [0.671, 0.856] | 1.072 [0.961, 1.185] |
| CAMF_C | 0.798 [0.703, 0.898] | 1.124 [1.020, 1.228] |
| BiasedMF | 0.810 [0.703, 0.920] | 1.198 [1.074, 1.320] |
| AutoInt | 0.811 [0.712, 0.913] | 1.149 [1.035, 1.267] |
| SVD++ | 0.823 [0.715, 0.935] | 1.213 [1.087, 1.336] |
| UAR | 0.834 [0.731, 0.942] | 1.196 [1.091, 1.301] |
| CAMF_CUCI | 0.860 [0.766, 0.958] | 1.159 [1.042, 1.278] |
| UserKNN | 0.870 [0.745, 1.001] | 1.343 [1.191, 1.492] |
| CAMF_CI | 0.880 [0.782, 0.984] | 1.203 [1.090, 1.316] |
| ItemKNN | 0.910 [0.802, 1.023] | 1.277 [1.153, 1.402] |
| GAR | 1.234 [1.140, 1.332] | 1.459 [1.351, 1.566] |
| IAR | 1.249 [1.152, 1.351] | 1.490 [1.378, 1.602] |

## YELP-WH (n=2000)

| model | MAE [95% CI] | RMSE [95% CI] |
|---|---|---|
| BiasedMF | 0.810 [0.782, 0.838] | 1.022 [0.986, 1.058] |
| DeepFM | 0.810 [0.782, 0.839] | 1.031 [0.994, 1.068] |
| AutoInt | 0.811 [0.783, 0.840] | 1.028 [0.992, 1.065] |
| NFM | 0.823 [0.796, 0.850] | 1.027 [0.992, 1.061] |
| UAR | 0.859 [0.828, 0.889] | 1.106 [1.064, 1.147] |
| mistral | 0.869 [0.841, 0.897] | 1.079 [1.042, 1.115] |
| IAR | 0.874 [0.846, 0.902] | 1.087 [1.051, 1.123] |
| llama | 0.876 [0.848, 0.904] | 1.085 [1.049, 1.122] |
| deepseek | 0.880 [0.852, 0.909] | 1.090 [1.054, 1.126] |
| eurollm | 0.887 [0.859, 0.915] | 1.090 [1.055, 1.125] |
| GAR | 0.928 [0.899, 0.957] | 1.136 [1.102, 1.169] |

## YELP-H (n=2000)

| model | MAE [95% CI] | RMSE [95% CI] |
|---|---|---|
| eurollm | 0.787 [0.760, 0.816] | 1.011 [0.971, 1.050] |
| mistral | 0.788 [0.760, 0.816] | 1.013 [0.973, 1.052] |
| deepseek | 0.791 [0.763, 0.819] | 1.014 [0.974, 1.053] |
| llama | 0.795 [0.767, 0.824] | 1.017 [0.978, 1.056] |
| BiasedMF | 0.810 [0.782, 0.838] | 1.022 [0.986, 1.058] |
| DeepFM | 0.810 [0.782, 0.839] | 1.031 [0.994, 1.068] |
| AutoInt | 0.811 [0.783, 0.840] | 1.028 [0.992, 1.065] |
| NFM | 0.823 [0.796, 0.850] | 1.027 [0.992, 1.061] |
| UAR | 0.859 [0.828, 0.889] | 1.106 [1.064, 1.147] |
| IAR | 0.874 [0.846, 0.902] | 1.087 [1.051, 1.123] |
| GAR | 0.928 [0.899, 0.957] | 1.136 [1.102, 1.169] |

## Paired comparisons (A - B; * = 95% interval excludes 0)

| comparison | A | B | ΔMAE [95% CI] p | ΔRMSE [95% CI] p |
|---|---|---|---|---|
| LLM vs best baseline (MAE) | mistral (LDOS-WH) | DeepFM (LDOS-WH) | +0.019 [-0.021, +0.058] p=0.348 | +0.007 [-0.034, +0.047] p=0.734 |
| LLM vs best baseline (MAE) | deepseek (LDOS-WH) | DeepFM (LDOS-WH) | +0.063 [+0.011, +0.117] p=0.019 * | +0.149 [+0.084, +0.218] p=0.000 * |
| LLM vs best baseline (MAE) | llama (LDOS-WH) | DeepFM (LDOS-WH) | +0.087 [+0.032, +0.143] p=0.003 * | +0.195 [+0.132, +0.259] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm (LDOS-WH) | DeepFM (LDOS-WH) | +0.087 [+0.035, +0.140] p=0.001 * | +0.162 [+0.107, +0.217] p=0.000 * |
| best LLM vs other LLM | mistral (LDOS-WH) | deepseek (LDOS-WH) | -0.044 [-0.106, +0.016] p=0.151 | -0.142 [-0.219, -0.069] p=0.000 * |
| best LLM vs other LLM | mistral (LDOS-WH) | eurollm (LDOS-WH) | -0.068 [-0.127, -0.007] p=0.028 * | -0.155 [-0.215, -0.093] p=0.000 * |
| best LLM vs other LLM | mistral (LDOS-WH) | llama (LDOS-WH) | -0.068 [-0.136, -0.002] p=0.045 * | -0.188 [-0.262, -0.116] p=0.000 * |
| LLM vs best baseline (MAE) | mistral (Depaul-WH) | CAMF_CU (Depaul-WH) | +0.531 [+0.458, +0.605] p=0.000 * | +0.441 [+0.361, +0.521] p=0.000 * |
| LLM vs best baseline (RMSE) | mistral (Depaul-WH) | FM (Depaul-WH) | +0.509 [+0.438, +0.583] p=0.000 * | +0.447 [+0.373, +0.524] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek (Depaul-WH) | CAMF_CU (Depaul-WH) | +0.442 [+0.371, +0.515] p=0.000 * | +0.383 [+0.302, +0.466] p=0.000 * |
| LLM vs best baseline (RMSE) | deepseek (Depaul-WH) | FM (Depaul-WH) | +0.420 [+0.352, +0.491] p=0.000 * | +0.389 [+0.313, +0.468] p=0.000 * |
| LLM vs best baseline (MAE) | llama (Depaul-WH) | CAMF_CU (Depaul-WH) | +0.446 [+0.375, +0.518] p=0.000 * | +0.389 [+0.308, +0.472] p=0.000 * |
| LLM vs best baseline (RMSE) | llama (Depaul-WH) | FM (Depaul-WH) | +0.424 [+0.356, +0.496] p=0.000 * | +0.395 [+0.319, +0.474] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm (Depaul-WH) | CAMF_CU (Depaul-WH) | +0.446 [+0.376, +0.518] p=0.000 * | +0.381 [+0.301, +0.463] p=0.000 * |
| LLM vs best baseline (RMSE) | eurollm (Depaul-WH) | FM (Depaul-WH) | +0.424 [+0.357, +0.494] p=0.000 * | +0.387 [+0.312, +0.465] p=0.000 * |
| best LLM vs other LLM | deepseek (Depaul-WH) | eurollm (Depaul-WH) | -0.004 [-0.013, +0.004] p=0.326 | +0.002 [-0.007, +0.010] p=0.705 |
| best LLM vs other LLM | deepseek (Depaul-WH) | llama (Depaul-WH) | -0.005 [-0.011, +0.002] p=0.183 | -0.006 [-0.013, +0.000] p=0.067 |
| best LLM vs other LLM | deepseek (Depaul-WH) | mistral (Depaul-WH) | -0.089 [-0.121, -0.056] p=0.000 * | -0.058 [-0.091, -0.024] p=0.000 * |
| LLM vs best baseline (MAE) | mistral (Tijuana-WH) | DeepFM (Tijuana-WH) | +0.247 [+0.144, +0.356] p=0.000 * | +0.369 [+0.238, +0.507] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek (Tijuana-WH) | DeepFM (Tijuana-WH) | +0.267 [+0.184, +0.353] p=0.000 * | +0.273 [+0.177, +0.374] p=0.000 * |
| LLM vs best baseline (MAE) | llama (Tijuana-WH) | DeepFM (Tijuana-WH) | +0.348 [+0.255, +0.444] p=0.000 * | +0.390 [+0.286, +0.501] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm (Tijuana-WH) | DeepFM (Tijuana-WH) | +0.313 [+0.229, +0.402] p=0.000 * | +0.293 [+0.188, +0.404] p=0.000 * |
| best LLM vs other LLM | mistral (Tijuana-WH) | deepseek (Tijuana-WH) | -0.019 [-0.095, +0.059] p=0.614 | +0.096 [-0.003, +0.200] p=0.059 |
| best LLM vs other LLM | mistral (Tijuana-WH) | eurollm (Tijuana-WH) | -0.066 [-0.146, +0.015] p=0.105 | +0.076 [-0.021, +0.174] p=0.129 |
| best LLM vs other LLM | mistral (Tijuana-WH) | llama (Tijuana-WH) | -0.101 [-0.192, -0.012] p=0.027 * | -0.021 [-0.137, +0.092] p=0.708 |
| LLM vs best baseline (MAE) | mistral (LDOS-H) | DeepFM (LDOS-H) | +0.175 [+0.112, +0.239] p=0.000 * | +0.265 [+0.195, +0.336] p=0.000 * |
| LLM vs best baseline (MAE) | mistral_clean (LDOS-H) | DeepFM (LDOS-H) | +0.097 [+0.039, +0.158] p=0.001 * | +0.225 [+0.159, +0.293] p=0.000 * |
| LLM vs best baseline (MAE) | mistral_noitclean (LDOS-H) | DeepFM (LDOS-H) | +0.088 [+0.029, +0.147] p=0.003 * | +0.206 [+0.138, +0.272] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek (LDOS-H) | DeepFM (LDOS-H) | +0.119 [+0.057, +0.182] p=0.000 * | +0.192 [+0.122, +0.263] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek_clean (LDOS-H) | DeepFM (LDOS-H) | +0.058 [+0.010, +0.108] p=0.018 * | +0.089 [+0.039, +0.141] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek_noitclean (LDOS-H) | DeepFM (LDOS-H) | +0.056 [+0.007, +0.107] p=0.021 * | +0.091 [+0.034, +0.151] p=0.001 * |
| LLM vs best baseline (MAE) | llama (LDOS-H) | DeepFM (LDOS-H) | +0.090 [+0.034, +0.146] p=0.001 * | +0.133 [+0.069, +0.199] p=0.000 * |
| LLM vs best baseline (MAE) | llama_clean (LDOS-H) | DeepFM (LDOS-H) | +0.077 [+0.027, +0.128] p=0.003 * | +0.107 [+0.051, +0.165] p=0.000 * |
| LLM vs best baseline (MAE) | llama_noitclean (LDOS-H) | DeepFM (LDOS-H) | +0.074 [+0.022, +0.125] p=0.006 * | +0.125 [+0.070, +0.179] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm (LDOS-H) | DeepFM (LDOS-H) | +0.071 [+0.016, +0.127] p=0.012 * | +0.116 [+0.059, +0.172] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm_clean (LDOS-H) | DeepFM (LDOS-H) | +0.082 [+0.033, +0.131] p=0.001 * | +0.123 [+0.070, +0.178] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm_noitclean (LDOS-H) | DeepFM (LDOS-H) | +0.115 [+0.071, +0.160] p=0.000 * | +0.092 [+0.045, +0.140] p=0.000 * |
| best LLM vs other LLM | deepseek_clean (LDOS-H) | llama_clean (LDOS-H) | -0.018 [-0.059, +0.021] p=0.373 | -0.017 [-0.061, +0.027] p=0.433 |
| best LLM vs other LLM | deepseek_clean (LDOS-H) | eurollm_clean (LDOS-H) | -0.023 [-0.066, +0.020] p=0.295 | -0.034 [-0.081, +0.014] p=0.156 |
| best LLM vs other LLM | deepseek_clean (LDOS-H) | mistral_clean (LDOS-H) | -0.039 [-0.098, +0.022] p=0.201 | -0.135 [-0.194, -0.076] p=0.000 * |
| leak-free retraining vs original training | mistral_clean (LDOS-H) | mistral (LDOS-H) | -0.078 [-0.160, +0.003] p=0.061 | -0.041 [-0.140, +0.058] p=0.413 |
| instruction-tuned vs base (leak-free training) | mistral_clean (LDOS-H) | mistral_noitclean (LDOS-H) | +0.009 [-0.036, +0.055] p=0.683 | +0.019 [-0.033, +0.072] p=0.478 |
| with vs without history | mistral (LDOS-H) | mistral (LDOS-WH) | +0.157 [+0.100, +0.215] p=0.000 * | +0.258 [+0.199, +0.318] p=0.000 * |
| with vs without history | mistral_clean (LDOS-H) | mistral (LDOS-WH) | +0.079 [+0.012, +0.146] p=0.019 * | +0.218 [+0.143, +0.294] p=0.000 * |
| leak-free retraining vs original training | deepseek_clean (LDOS-H) | deepseek (LDOS-H) | -0.060 [-0.119, -0.002] p=0.041 * | -0.103 [-0.170, -0.037] p=0.002 * |
| instruction-tuned vs base (leak-free training) | deepseek_clean (LDOS-H) | deepseek_noitclean (LDOS-H) | +0.002 [-0.032, +0.036] p=0.889 | -0.001 [-0.037, +0.034] p=0.955 |
| with vs without history | deepseek (LDOS-H) | deepseek (LDOS-WH) | +0.056 [-0.016, +0.129] p=0.135 | +0.043 [-0.043, +0.131] p=0.342 |
| with vs without history | deepseek_clean (LDOS-H) | deepseek (LDOS-WH) | -0.005 [-0.062, +0.052] p=0.867 | -0.060 [-0.121, +0.001] p=0.054 |
| leak-free retraining vs original training | llama_clean (LDOS-H) | llama (LDOS-H) | -0.013 [-0.056, +0.030] p=0.547 | -0.027 [-0.081, +0.027] p=0.332 |
| instruction-tuned vs base (leak-free training) | llama_clean (LDOS-H) | llama_noitclean (LDOS-H) | +0.003 [-0.033, +0.039] p=0.878 | -0.018 [-0.062, +0.030] p=0.449 |
| with vs without history | llama (LDOS-H) | llama (LDOS-WH) | +0.003 [-0.055, +0.063] p=0.900 | -0.061 [-0.123, +0.004] p=0.067 |
| with vs without history | llama_clean (LDOS-H) | llama (LDOS-WH) | -0.010 [-0.071, +0.052] p=0.753 | -0.088 [-0.156, -0.020] p=0.011 * |
| leak-free retraining vs original training | eurollm_clean (LDOS-H) | eurollm (LDOS-H) | +0.010 [-0.040, +0.059] p=0.697 | +0.007 [-0.045, +0.060] p=0.799 |
| instruction-tuned vs base (leak-free training) | eurollm_clean (LDOS-H) | eurollm_noitclean (LDOS-H) | -0.033 [-0.090, +0.022] p=0.242 | +0.032 [-0.028, +0.092] p=0.317 |
| with vs without history | eurollm (LDOS-H) | eurollm (LDOS-WH) | -0.015 [-0.076, +0.047] p=0.630 | -0.046 [-0.113, +0.019] p=0.185 |
| with vs without history | eurollm_clean (LDOS-H) | eurollm (LDOS-WH) | -0.005 [-0.057, +0.047] p=0.860 | -0.038 [-0.097, +0.020] p=0.210 |
| LLM vs best baseline (MAE) | mistral (Depaul-H) | CAMF_CU (Depaul-H) | +0.108 [+0.057, +0.162] p=0.000 * | +0.177 [+0.105, +0.251] p=0.000 * |
| LLM vs best baseline (RMSE) | mistral (Depaul-H) | FM (Depaul-H) | +0.086 [+0.034, +0.142] p=0.000 * | +0.183 [+0.115, +0.256] p=0.000 * |
| LLM vs best baseline (MAE) | mistral_clean (Depaul-H) | CAMF_CU (Depaul-H) | +0.083 [+0.025, +0.141] p=0.005 * | +0.198 [+0.120, +0.277] p=0.000 * |
| LLM vs best baseline (RMSE) | mistral_clean (Depaul-H) | FM (Depaul-H) | +0.061 [+0.000, +0.123] p=0.048 * | +0.204 [+0.126, +0.283] p=0.000 * |
| LLM vs best baseline (MAE) | mistral_noitclean (Depaul-H) | CAMF_CU (Depaul-H) | +0.017 [-0.037, +0.076] p=0.542 | +0.127 [+0.056, +0.201] p=0.000 * |
| LLM vs best baseline (RMSE) | mistral_noitclean (Depaul-H) | FM (Depaul-H) | -0.005 [-0.060, +0.052] p=0.883 | +0.134 [+0.061, +0.208] p=0.001 * |
| LLM vs best baseline (MAE) | deepseek (Depaul-H) | CAMF_CU (Depaul-H) | +0.103 [+0.050, +0.157] p=0.000 * | +0.150 [+0.080, +0.222] p=0.000 * |
| LLM vs best baseline (RMSE) | deepseek (Depaul-H) | FM (Depaul-H) | +0.081 [+0.028, +0.135] p=0.004 * | +0.157 [+0.090, +0.225] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek_clean (Depaul-H) | CAMF_CU (Depaul-H) | +0.086 [+0.036, +0.138] p=0.000 * | +0.126 [+0.061, +0.193] p=0.000 * |
| LLM vs best baseline (RMSE) | deepseek_clean (Depaul-H) | FM (Depaul-H) | +0.064 [+0.013, +0.118] p=0.017 * | +0.132 [+0.068, +0.198] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek_noitclean (Depaul-H) | CAMF_CU (Depaul-H) | +0.086 [+0.034, +0.139] p=0.001 * | +0.118 [+0.048, +0.191] p=0.001 * |
| LLM vs best baseline (RMSE) | deepseek_noitclean (Depaul-H) | FM (Depaul-H) | +0.064 [+0.010, +0.119] p=0.019 * | +0.125 [+0.057, +0.195] p=0.000 * |
| LLM vs best baseline (MAE) | llama (Depaul-H) | CAMF_CU (Depaul-H) | +0.071 [+0.015, +0.126] p=0.012 * | +0.135 [+0.055, +0.215] p=0.001 * |
| LLM vs best baseline (RMSE) | llama (Depaul-H) | FM (Depaul-H) | +0.049 [-0.007, +0.106] p=0.085 | +0.141 [+0.064, +0.220] p=0.000 * |
| LLM vs best baseline (MAE) | llama_clean (Depaul-H) | CAMF_CU (Depaul-H) | +0.028 [-0.025, +0.082] p=0.310 | +0.109 [+0.032, +0.188] p=0.004 * |
| LLM vs best baseline (RMSE) | llama_clean (Depaul-H) | FM (Depaul-H) | +0.006 [-0.047, +0.062] p=0.840 | +0.115 [+0.041, +0.194] p=0.002 * |
| LLM vs best baseline (MAE) | llama_noitclean (Depaul-H) | CAMF_CU (Depaul-H) | +0.033 [-0.019, +0.088] p=0.214 | +0.122 [+0.046, +0.198] p=0.001 * |
| LLM vs best baseline (RMSE) | llama_noitclean (Depaul-H) | FM (Depaul-H) | +0.011 [-0.044, +0.069] p=0.695 | +0.128 [+0.051, +0.210] p=0.001 * |
| LLM vs best baseline (MAE) | eurollm (Depaul-H) | CAMF_CU (Depaul-H) | +0.154 [+0.096, +0.212] p=0.000 * | +0.224 [+0.148, +0.303] p=0.000 * |
| LLM vs best baseline (RMSE) | eurollm (Depaul-H) | FM (Depaul-H) | +0.132 [+0.076, +0.191] p=0.000 * | +0.231 [+0.159, +0.306] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm_clean (Depaul-H) | CAMF_CU (Depaul-H) | +0.023 [-0.028, +0.076] p=0.374 | +0.059 [-0.010, +0.130] p=0.099 |
| LLM vs best baseline (RMSE) | eurollm_clean (Depaul-H) | FM (Depaul-H) | +0.001 [-0.050, +0.052] p=0.956 | +0.065 [+0.001, +0.134] p=0.047 * |
| LLM vs best baseline (MAE) | eurollm_noitclean (Depaul-H) | CAMF_CU (Depaul-H) | +0.093 [+0.038, +0.149] p=0.000 * | +0.153 [+0.074, +0.233] p=0.000 * |
| LLM vs best baseline (RMSE) | eurollm_noitclean (Depaul-H) | FM (Depaul-H) | +0.071 [+0.016, +0.127] p=0.008 * | +0.159 [+0.081, +0.239] p=0.000 * |
| best LLM vs other LLM | eurollm_clean (Depaul-H) | llama_clean (Depaul-H) | -0.005 [-0.050, +0.040] p=0.843 | -0.050 [-0.112, +0.011] p=0.111 |
| best LLM vs other LLM | eurollm_clean (Depaul-H) | mistral_clean (Depaul-H) | -0.060 [-0.113, -0.007] p=0.025 * | -0.139 [-0.204, -0.073] p=0.000 * |
| best LLM vs other LLM | eurollm_clean (Depaul-H) | deepseek_clean (Depaul-H) | -0.064 [-0.113, -0.013] p=0.014 * | -0.067 [-0.133, +0.001] p=0.055 |
| leak-free retraining vs original training | mistral_clean (Depaul-H) | mistral (Depaul-H) | -0.026 [-0.084, +0.031] p=0.367 | +0.021 [-0.057, +0.096] p=0.598 |
| instruction-tuned vs base (leak-free training) | mistral_clean (Depaul-H) | mistral_noitclean (Depaul-H) | +0.065 [+0.008, +0.122] p=0.024 * | +0.070 [-0.008, +0.149] p=0.083 |
| with vs without history | mistral (Depaul-H) | mistral (Depaul-WH) | -0.422 [-0.500, -0.345] p=0.000 * | -0.264 [-0.347, -0.181] p=0.000 * |
| with vs without history | mistral_clean (Depaul-H) | mistral (Depaul-WH) | -0.448 [-0.532, -0.367] p=0.000 * | -0.243 [-0.336, -0.155] p=0.000 * |
| leak-free retraining vs original training | deepseek_clean (Depaul-H) | deepseek (Depaul-H) | -0.017 [-0.070, +0.037] p=0.528 | -0.024 [-0.094, +0.044] p=0.480 |
| instruction-tuned vs base (leak-free training) | deepseek_clean (Depaul-H) | deepseek_noitclean (Depaul-H) | +0.000 [-0.047, +0.048] p=0.997 | +0.007 [-0.054, +0.067] p=0.812 |
| with vs without history | deepseek (Depaul-H) | deepseek (Depaul-WH) | -0.339 [-0.412, -0.267] p=0.000 * | -0.232 [-0.315, -0.151] p=0.000 * |
| with vs without history | deepseek_clean (Depaul-H) | deepseek (Depaul-WH) | -0.356 [-0.429, -0.286] p=0.000 * | -0.257 [-0.334, -0.179] p=0.000 * |
| leak-free retraining vs original training | llama_clean (Depaul-H) | llama (Depaul-H) | -0.043 [-0.091, +0.005] p=0.078 | -0.026 [-0.090, +0.039] p=0.429 |
| instruction-tuned vs base (leak-free training) | llama_clean (Depaul-H) | llama_noitclean (Depaul-H) | -0.006 [-0.051, +0.041] p=0.808 | -0.013 [-0.077, +0.054] p=0.686 |
| with vs without history | llama (Depaul-H) | llama (Depaul-WH) | -0.376 [-0.451, -0.301] p=0.000 * | -0.254 [-0.345, -0.164] p=0.000 * |
| with vs without history | llama_clean (Depaul-H) | llama (Depaul-WH) | -0.419 [-0.495, -0.344] p=0.000 * | -0.280 [-0.368, -0.192] p=0.000 * |
| leak-free retraining vs original training | eurollm_clean (Depaul-H) | eurollm (Depaul-H) | -0.131 [-0.184, -0.079] p=0.000 * | -0.165 [-0.229, -0.101] p=0.000 * |
| instruction-tuned vs base (leak-free training) | eurollm_clean (Depaul-H) | eurollm_noitclean (Depaul-H) | -0.070 [-0.118, -0.021] p=0.004 * | -0.094 [-0.160, -0.027] p=0.004 * |
| with vs without history | eurollm (Depaul-H) | eurollm (Depaul-WH) | -0.293 [-0.367, -0.220] p=0.000 * | -0.157 [-0.241, -0.076] p=0.000 * |
| with vs without history | eurollm_clean (Depaul-H) | eurollm (Depaul-WH) | -0.424 [-0.495, -0.352] p=0.000 * | -0.322 [-0.403, -0.242] p=0.000 * |
| LLM vs best baseline (MAE) | mistral (Restaurant-H) | DeepFM (Restaurant-H) | +0.076 [-0.002, +0.157] p=0.055 | +0.128 [+0.023, +0.234] p=0.016 * |
| LLM vs best baseline (MAE) | mistral_clean (Restaurant-H) | DeepFM (Restaurant-H) | +0.100 [+0.015, +0.188] p=0.019 * | +0.212 [+0.103, +0.326] p=0.000 * |
| LLM vs best baseline (MAE) | mistral_noit (Restaurant-H) | DeepFM (Restaurant-H) | +0.142 [+0.059, +0.227] p=0.001 * | +0.220 [+0.106, +0.342] p=0.000 * |
| LLM vs best baseline (MAE) | mistral_noitclean (Restaurant-H) | DeepFM (Restaurant-H) | +0.123 [+0.031, +0.216] p=0.011 * | +0.250 [+0.131, +0.369] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek (Restaurant-H) | DeepFM (Restaurant-H) | +0.140 [+0.063, +0.221] p=0.000 * | +0.197 [+0.104, +0.299] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek_clean (Restaurant-H) | DeepFM (Restaurant-H) | +0.178 [+0.094, +0.267] p=0.000 * | +0.284 [+0.180, +0.395] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek_noitclean (Restaurant-H) | DeepFM (Restaurant-H) | +0.212 [+0.127, +0.300] p=0.000 * | +0.285 [+0.189, +0.387] p=0.000 * |
| LLM vs best baseline (MAE) | llama (Restaurant-H) | DeepFM (Restaurant-H) | +0.206 [+0.115, +0.298] p=0.000 * | +0.294 [+0.182, +0.412] p=0.000 * |
| LLM vs best baseline (MAE) | llama_clean (Restaurant-H) | DeepFM (Restaurant-H) | +0.129 [+0.050, +0.212] p=0.001 * | +0.221 [+0.125, +0.324] p=0.000 * |
| LLM vs best baseline (MAE) | llama_noitclean (Restaurant-H) | DeepFM (Restaurant-H) | +0.156 [+0.077, +0.238] p=0.000 * | +0.203 [+0.114, +0.299] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm (Restaurant-H) | DeepFM (Restaurant-H) | +0.162 [+0.073, +0.253] p=0.000 * | +0.197 [+0.072, +0.330] p=0.003 * |
| LLM vs best baseline (MAE) | eurollm_clean (Restaurant-H) | DeepFM (Restaurant-H) | +0.114 [+0.036, +0.196] p=0.005 * | +0.191 [+0.092, +0.292] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm_noitclean (Restaurant-H) | DeepFM (Restaurant-H) | +0.147 [+0.065, +0.233] p=0.001 * | +0.243 [+0.138, +0.350] p=0.000 * |
| best LLM vs other LLM | mistral_clean (Restaurant-H) | eurollm_clean (Restaurant-H) | -0.014 [-0.077, +0.050] p=0.656 | +0.021 [-0.072, +0.124] p=0.684 |
| best LLM vs other LLM | mistral_clean (Restaurant-H) | llama_clean (Restaurant-H) | -0.029 [-0.096, +0.039] p=0.385 | -0.010 [-0.112, +0.092] p=0.843 |
| best LLM vs other LLM | mistral_clean (Restaurant-H) | deepseek_clean (Restaurant-H) | -0.077 [-0.144, -0.011] p=0.022 * | -0.072 [-0.165, +0.023] p=0.136 |
| leak-free retraining vs original training | mistral_clean (Restaurant-H) | mistral (Restaurant-H) | +0.024 [-0.041, +0.089] p=0.467 | +0.084 [-0.016, +0.189] p=0.101 |
| instruction-tuned vs base | mistral (Restaurant-H) | mistral_noit (Restaurant-H) | -0.065 [-0.124, -0.014] p=0.012 * | -0.092 [-0.190, -0.008] p=0.031 * |
| instruction-tuned vs base (leak-free training) | mistral_clean (Restaurant-H) | mistral_noitclean (Restaurant-H) | -0.023 [-0.089, +0.043] p=0.503 | -0.038 [-0.138, +0.064] p=0.458 |
| with vs without history | mistral (Restaurant-H) | mistral (Tijuana-WH) | -0.171 [-0.271, -0.081] p=0.000 * | -0.241 [-0.361, -0.127] p=0.000 * |
| with vs without history | mistral_clean (Restaurant-H) | mistral (Tijuana-WH) | -0.147 [-0.250, -0.049] p=0.003 * | -0.157 [-0.287, -0.035] p=0.013 * |
| leak-free retraining vs original training | deepseek_clean (Restaurant-H) | deepseek (Restaurant-H) | +0.037 [+0.000, +0.074] p=0.048 * | +0.087 [+0.040, +0.132] p=0.000 * |
| instruction-tuned vs base (leak-free training) | deepseek_clean (Restaurant-H) | deepseek_noitclean (Restaurant-H) | -0.034 [-0.074, +0.006] p=0.089 | -0.001 [-0.048, +0.044] p=0.964 |
| with vs without history | deepseek (Restaurant-H) | deepseek (Tijuana-WH) | -0.126 [-0.190, -0.062] p=0.000 * | -0.076 [-0.146, -0.004] p=0.038 * |
| with vs without history | deepseek_clean (Restaurant-H) | deepseek (Tijuana-WH) | -0.089 [-0.157, -0.021] p=0.011 * | +0.011 [-0.064, +0.086] p=0.770 |
| leak-free retraining vs original training | llama_clean (Restaurant-H) | llama (Restaurant-H) | -0.077 [-0.137, -0.018] p=0.010 * | -0.073 [-0.158, +0.012] p=0.096 |
| instruction-tuned vs base (leak-free training) | llama_clean (Restaurant-H) | llama_noitclean (Restaurant-H) | -0.027 [-0.078, +0.024] p=0.309 | +0.018 [-0.059, +0.099] p=0.669 |
| with vs without history | llama (Restaurant-H) | llama (Tijuana-WH) | -0.142 [-0.233, -0.052] p=0.001 * | -0.096 [-0.209, +0.013] p=0.086 |
| with vs without history | llama_clean (Restaurant-H) | llama (Tijuana-WH) | -0.220 [-0.302, -0.139] p=0.000 * | -0.169 [-0.276, -0.065] p=0.002 * |
| leak-free retraining vs original training | eurollm_clean (Restaurant-H) | eurollm (Restaurant-H) | -0.047 [-0.110, +0.013] p=0.128 | -0.006 [-0.104, +0.082] p=0.906 |
| instruction-tuned vs base (leak-free training) | eurollm_clean (Restaurant-H) | eurollm_noitclean (Restaurant-H) | -0.033 [-0.082, +0.015] p=0.179 | -0.052 [-0.123, +0.016] p=0.138 |
| with vs without history | eurollm (Restaurant-H) | eurollm (Tijuana-WH) | -0.152 [-0.224, -0.080] p=0.000 * | -0.096 [-0.192, -0.000] p=0.049 * |
| with vs without history | eurollm_clean (Restaurant-H) | eurollm (Tijuana-WH) | -0.199 [-0.275, -0.125] p=0.000 * | -0.102 [-0.201, -0.011] p=0.029 * |
| LLM vs best baseline (MAE) | mistral (YELP-WH) | BiasedMF (YELP-WH) | +0.059 [+0.042, +0.076] p=0.000 * | +0.057 [+0.036, +0.077] p=0.000 * |
| LLM vs best baseline (MAE) | deepseek (YELP-WH) | BiasedMF (YELP-WH) | +0.070 [+0.053, +0.088] p=0.000 * | +0.068 [+0.048, +0.087] p=0.000 * |
| LLM vs best baseline (MAE) | llama (YELP-WH) | BiasedMF (YELP-WH) | +0.066 [+0.049, +0.083] p=0.000 * | +0.063 [+0.043, +0.083] p=0.000 * |
| LLM vs best baseline (MAE) | eurollm (YELP-WH) | BiasedMF (YELP-WH) | +0.077 [+0.060, +0.095] p=0.000 * | +0.068 [+0.048, +0.088] p=0.000 * |
| best LLM vs other LLM | mistral (YELP-WH) | llama (YELP-WH) | -0.007 [-0.014, -0.000] p=0.038 * | -0.007 [-0.014, +0.001] p=0.085 |
| best LLM vs other LLM | mistral (YELP-WH) | deepseek (YELP-WH) | -0.011 [-0.019, -0.004] p=0.001 * | -0.011 [-0.020, -0.003] p=0.008 * |
| best LLM vs other LLM | mistral (YELP-WH) | eurollm (YELP-WH) | -0.018 [-0.025, -0.011] p=0.000 * | -0.011 [-0.020, -0.003] p=0.009 * |
| LLM vs best baseline (MAE) | mistral (YELP-H) | BiasedMF (YELP-H) | -0.022 [-0.039, -0.005] p=0.012 * | -0.009 [-0.029, +0.011] p=0.376 |
| LLM vs best baseline (MAE) | deepseek (YELP-H) | BiasedMF (YELP-H) | -0.019 [-0.036, -0.002] p=0.027 * | -0.009 [-0.029, +0.012] p=0.400 |
| LLM vs best baseline (MAE) | llama (YELP-H) | BiasedMF (YELP-H) | -0.015 [-0.031, +0.002] p=0.082 | -0.005 [-0.025, +0.014] p=0.589 |
| LLM vs best baseline (MAE) | eurollm (YELP-H) | BiasedMF (YELP-H) | -0.022 [-0.039, -0.006] p=0.010 * | -0.011 [-0.031, +0.008] p=0.270 |
| best LLM vs other LLM | eurollm (YELP-H) | mistral (YELP-H) | -0.001 [-0.009, +0.008] p=0.907 | -0.002 [-0.012, +0.008] p=0.693 |
| best LLM vs other LLM | eurollm (YELP-H) | deepseek (YELP-H) | -0.003 [-0.011, +0.005] p=0.453 | -0.002 [-0.011, +0.007] p=0.600 |
| best LLM vs other LLM | eurollm (YELP-H) | llama (YELP-H) | -0.008 [-0.016, +0.000] p=0.056 | -0.006 [-0.015, +0.003] p=0.202 |
| with vs without history | mistral (YELP-H) | mistral (YELP-WH) | -0.081 [-0.097, -0.065] p=0.000 * | -0.066 [-0.087, -0.046] p=0.000 * |
| with vs without history | deepseek (YELP-H) | deepseek (YELP-WH) | -0.090 [-0.106, -0.074] p=0.000 * | -0.077 [-0.096, -0.058] p=0.000 * |
| with vs without history | llama (YELP-H) | llama (YELP-WH) | -0.080 [-0.096, -0.065] p=0.000 * | -0.068 [-0.087, -0.051] p=0.000 * |
| with vs without history | eurollm (YELP-H) | eurollm (YELP-WH) | -0.099 [-0.115, -0.084] p=0.000 * | -0.079 [-0.098, -0.061] p=0.000 * |
