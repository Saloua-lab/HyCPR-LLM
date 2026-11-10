# BMA over the fusion pool vs convex fusion (test; * = paired bootstrap 95% CI excludes 0)

| LLM setting | n val | convex MAE/RMSE | BMA (vs convex) | top BMA weights | pseudo-BMA+ (vs convex) | top pseudo-BMA+ weights |
|---|---|---|---|---|---|---|
| mistral LDOS-WH | 144 | 0.756/0.986 | 0.751/0.940 (-0.005, -0.046*) | CAMF_CI 0.34, CAMF_C 0.20, BiasedMF 0.18 | 0.743/0.939 (-0.013, -0.047*) | CAMF_CI 0.22, LLM 0.12, CAMF_C 0.11 |
| eurollm LDOS-WH | 144 | 0.775/1.013 | 0.827/1.113 (+0.053*, +0.100*) | LLM 0.98, CAMF_CI 0.01, CAMF_C 0.00 | 0.777/1.009 (+0.002, -0.004) | LLM 0.64, CAMF_CI 0.08, CAMF_C 0.05 |
| llama LDOS-WH | 144 | 0.763/0.986 | 0.751/0.940 (-0.012, -0.045*) | CAMF_CI 0.34, CAMF_C 0.20, BiasedMF 0.18 | 0.745/0.936 (-0.018, -0.050*) | CAMF_CI 0.23, CAMF_C 0.12, CAMF_CUCI 0.11 |
| deepseek LDOS-WH | 144 | 0.754/0.979 | 0.750/0.971 (-0.004, -0.008) | LLM 0.54, CAMF_CI 0.16, CAMF_C 0.09 | 0.743/0.957 (-0.011, -0.022*) | LLM 0.46, CAMF_CI 0.12, CAMF_CUCI 0.08 |
| mistral Depaul-WH | 249 | 0.767/0.982 | 0.744/0.986 (-0.023*, +0.004) | NFM 1.00, CAMF_CU 0.00, FM 0.00 | 0.727/0.969 (-0.040*, -0.012*) | NFM 0.70, CAMF_CU 0.17, FM 0.08 |
| eurollm Depaul-WH | 249 | 0.767/0.981 | 0.744/0.986 (-0.023*, +0.004) | NFM 1.00, CAMF_CU 0.00, FM 0.00 | 0.727/0.969 (-0.040*, -0.012*) | NFM 0.70, CAMF_CU 0.17, FM 0.08 |
| llama Depaul-WH | 249 | 0.767/0.982 | 0.744/0.986 (-0.023*, +0.004) | NFM 1.00, CAMF_CU 0.00, FM 0.00 | 0.727/0.969 (-0.040*, -0.013*) | NFM 0.70, CAMF_CU 0.17, FM 0.08 |
| deepseek Depaul-WH | 249 | 0.781/0.983 | 0.744/0.986 (-0.037*, +0.003) | NFM 1.00, CAMF_CU 0.00, FM 0.00 | 0.727/0.969 (-0.054*, -0.013*) | NFM 0.70, CAMF_CU 0.17, FM 0.08 |
| mistral Tijuana-WH | 92 | 0.533/0.822 | 0.498/0.824 (-0.036*, +0.003) | DeepFM 1.00, LLM 0.00, NFM 0.00 | 0.531/0.822 (-0.002, +0.000) | DeepFM 0.83, AutoInt 0.08, LLM 0.07 |
| eurollm Tijuana-WH | 92 | 0.580/0.866 | 0.537/0.828 (-0.043*, -0.038*) | DeepFM 0.75, LLM 0.25, NFM 0.00 | 0.578/0.863 (-0.001, -0.003*) | DeepFM 0.56, LLM 0.43, AutoInt 0.01 |
| llama Tijuana-WH | 92 | 0.557/0.841 | 0.498/0.824 (-0.059*, -0.017) | DeepFM 1.00, LLM 0.00, NFM 0.00 | 0.537/0.827 (-0.020*, -0.014) | DeepFM 0.83, LLM 0.08, AutoInt 0.06 |
| deepseek Tijuana-WH | 92 | 0.498/0.824 | 0.498/0.824 (+0.000*, +0.000) | DeepFM 1.00, NFM 0.00, CAMF_CU 0.00 | 0.531/0.826 (+0.033*, +0.002) | DeepFM 0.86, AutoInt 0.10, CAMF_CU 0.03 |
| mistral LDOS-H | 144 | 0.711/0.920 | 0.753/0.992 (+0.042*, +0.071*) | LLM 0.94, CAMF_CI 0.02, CAMF_C 0.01 | 0.710/0.920 (-0.001, -0.000) | LLM 0.59, CAMF_CI 0.08, CAMF_CUCI 0.06 |
| eurollm LDOS-H | 144 | 0.768/0.976 | 0.823/1.058 (+0.056*, +0.082*) | LLM 1.00, CAMF_CI 0.00, CAMF_C 0.00 | 0.784/1.003 (+0.017*, +0.027*) | LLM 0.80, CAMF_CI 0.03, CAMF_C 0.03 |
| llama LDOS-H | 144 | 0.780/0.977 | 0.848/1.058 (+0.068*, +0.081*) | LLM 0.99, CAMF_CI 0.00, CAMF_C 0.00 | 0.792/0.982 (+0.012*, +0.006) | LLM 0.71, CAMF_CI 0.06, CAMF_C 0.04 |
| deepseek LDOS-H | 144 | 0.749/0.948 | 0.806/1.028 (+0.057*, +0.080*) | LLM 0.98, CAMF_CI 0.01, CAMF_C 0.00 | 0.753/0.947 (+0.003, -0.001) | LLM 0.64, CAMF_C 0.06, CAMF_CI 0.06 |
| mistral Depaul-H | 249 | 0.711/0.999 | 0.734/0.972 (+0.023, -0.026) | NFM 0.81, LLM 0.19, CAMF_CU 0.00 | 0.721/0.973 (+0.009, -0.025*) | NFM 0.43, LLM 0.39, CAMF_CU 0.09 |
| eurollm Depaul-H | 249 | 0.749/0.983 | 0.744/0.986 (-0.005, +0.003) | NFM 1.00, CAMF_CU 0.00, FM 0.00 | 0.727/0.969 (-0.022*, -0.014) | NFM 0.70, CAMF_CU 0.17, FM 0.08 |
| llama Depaul-H | 249 | 0.725/0.960 | 0.744/0.986 (+0.019*, +0.025*) | NFM 1.00, CAMF_CU 0.00, FM 0.00 | 0.726/0.967 (+0.001, +0.007) | NFM 0.69, CAMF_CU 0.17, FM 0.08 |
| deepseek Depaul-H | 249 | 0.721/0.982 | 0.744/0.986 (+0.024, +0.003) | NFM 1.00, CAMF_CU 0.00, FM 0.00 | 0.726/0.967 (+0.006, -0.016) | NFM 0.68, CAMF_CU 0.16, FM 0.08 |
| mistral Restaurant-H | 92 | 0.523/0.840 | 0.499/0.823 (-0.024, -0.018) | DeepFM 0.98, LLM 0.02, NFM 0.00 | 0.524/0.832 (+0.002, -0.008) | DeepFM 0.66, LLM 0.29, AutoInt 0.03 |
| eurollm Restaurant-H | 92 | 0.505/0.820 | 0.498/0.824 (-0.008, +0.004) | DeepFM 1.00, NFM 0.00, CAMF_CU 0.00 | 0.530/0.826 (+0.025*, +0.006) | DeepFM 0.85, AutoInt 0.10, CAMF_CU 0.03 |
| llama Restaurant-H | 92 | 0.522/0.822 | 0.498/0.824 (-0.025*, +0.002) | DeepFM 1.00, NFM 0.00, LLM 0.00 | 0.530/0.825 (+0.008, +0.003) | DeepFM 0.85, AutoInt 0.09, LLM 0.03 |
| deepseek Restaurant-H | 92 | 0.531/0.832 | 0.498/0.824 (-0.033*, -0.008) | DeepFM 1.00, LLM 0.00, NFM 0.00 | 0.534/0.828 (+0.003, -0.004) | DeepFM 0.79, LLM 0.12, AutoInt 0.06 |
| mistral MovieLens-H | 2000 | 0.687/0.869 | 0.697/0.883 (+0.010*, +0.014*) | LLM 1.00, DeepFM 0.00, AutoInt 0.00 | 0.697/0.883 (+0.010*, +0.014*) | LLM 1.00, AutoInt 0.00, DeepFM 0.00 |
| eurollm MovieLens-H | 2000 | 0.686/0.874 | 0.695/0.889 (+0.009*, +0.015*) | LLM 1.00, DeepFM 0.00, AutoInt 0.00 | 0.695/0.889 (+0.009*, +0.015*) | LLM 1.00, DeepFM 0.00, AutoInt 0.00 |
| llama MovieLens-H | 2000 | 0.686/0.873 | 0.696/0.887 (+0.009*, +0.014*) | LLM 1.00, DeepFM 0.00, AutoInt 0.00 | 0.696/0.887 (+0.009*, +0.014*) | LLM 1.00, DeepFM 0.00, AutoInt 0.00 |
| deepseek MovieLens-H | 2000 | 0.686/0.874 | 0.696/0.889 (+0.011*, +0.015*) | LLM 1.00, DeepFM 0.00, AutoInt 0.00 | 0.696/0.889 (+0.011*, +0.015*) | LLM 1.00, DeepFM 0.00, AutoInt 0.00 |
| mistral AmazonBooks-H | 2000 | 0.545/0.771 | 0.555/0.780 (+0.010*, +0.009*) | LLM 1.00, MLP 0.00, DeepFM 0.00 | 0.553/0.778 (+0.008*, +0.007) | LLM 0.95, MLP 0.04, DeepFM 0.01 |
| eurollm AmazonBooks-H | 2000 | 0.555/0.781 | 0.578/0.798 (+0.023*, +0.016*) | LLM 0.99, MLP 0.01, DeepFM 0.00 | 0.559/0.781 (+0.004*, -0.000) | LLM 0.57, MLP 0.37, DeepFM 0.05 |
| llama AmazonBooks-H | 2000 | 0.549/0.774 | 0.563/0.786 (+0.014*, +0.012*) | LLM 1.00, MLP 0.00, DeepFM 0.00 | 0.558/0.781 (+0.010*, +0.006*) | LLM 0.89, MLP 0.09, DeepFM 0.01 |
| deepseek AmazonBooks-H | 2000 | 0.552/0.777 | 0.569/0.791 (+0.017*, +0.013*) | LLM 1.00, MLP 0.00, DeepFM 0.00 | 0.556/0.778 (+0.004*, +0.000) | LLM 0.67, MLP 0.28, DeepFM 0.04 |