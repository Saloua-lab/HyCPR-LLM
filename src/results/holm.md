# Holm-Bonferroni correction of the main claims

## F1: 64 tests, significant before / after Holm: 23 / 14
- lost: eurollm LDOS-WH rmse vs DeepFM (+0.066, p=0.0022)
- lost: mistral Tijuana-WH mae vs DeepFM (+0.036, p=0.0056)
- lost: eurollm Depaul-H mae vs CAMF_CU (+0.036, p=0.0068)
- lost: llama Restaurant-H mae vs DeepFM (+0.025, p=0.0100)
- lost: deepseek Restaurant-H mae vs DeepFM (+0.033, p=0.0028)
- lost: mistral YELP-WH mae vs BiasedMF (+0.006, p=0.0202)
- lost: eurollm YELP-WH mae vs BiasedMF (+0.008, p=0.0032)
- lost: llama YELP-WH mae vs BiasedMF (+0.008, p=0.0028)
- lost: deepseek YELP-WH mae vs BiasedMF (+0.008, p=0.0040)

| LLM | setting | metric | vs | diff [95% CI] | rel. % | p | p Holm |
|---|---|---|---|---|---|---|---|
| mistral | LDOS-WH | mae | DeepFM | +0.0109 [-0.025, +0.048] | +1.5 | 0.5670 | 1.0000 |
| mistral | LDOS-WH | rmse | DeepFM | +0.0388 [-0.000, +0.079] | +4.1 | 0.0508 | 1.0000 |
| eurollm | LDOS-WH | mae | DeepFM | +0.0297 [-0.008, +0.068] | +4.0 | 0.1258 | 1.0000 |
| eurollm | LDOS-WH | rmse | DeepFM | +0.0658 [+0.021, +0.111] | +6.9 | 0.0022 | 0.1100 |
| llama | LDOS-WH | mae | DeepFM | +0.0179 [-0.014, +0.050] | +2.4 | 0.2670 | 1.0000 |
| llama | LDOS-WH | rmse | DeepFM | +0.0380 [-0.000, +0.077] | +4.0 | 0.0508 | 1.0000 |
| deepseek | LDOS-WH | mae | DeepFM | +0.0087 [-0.025, +0.044] | +1.2 | 0.6224 | 1.0000 |
| deepseek | LDOS-WH | rmse | DeepFM | +0.0318 [-0.009, +0.074] | +3.4 | 0.1194 | 1.0000 |
| mistral | Depaul-WH | mae | CAMF_CU | +0.0547 [+0.027, +0.083] | +7.7 | 0.0001 | 0.0064 |
| mistral | Depaul-WH | rmse | FM | -0.0041 [-0.031, +0.024] | -0.4 | 0.7828 | 1.0000 |
| eurollm | Depaul-WH | mae | CAMF_CU | +0.0551 [+0.028, +0.083] | +7.7 | 0.0001 | 0.0064 |
| eurollm | Depaul-WH | rmse | FM | -0.0044 [-0.031, +0.023] | -0.4 | 0.7650 | 1.0000 |
| llama | Depaul-WH | mae | CAMF_CU | +0.0551 [+0.028, +0.083] | +7.7 | 0.0001 | 0.0064 |
| llama | Depaul-WH | rmse | FM | -0.0039 [-0.030, +0.024] | -0.4 | 0.7918 | 1.0000 |
| deepseek | Depaul-WH | mae | CAMF_CU | +0.0684 [+0.040, +0.097] | +9.6 | 0.0001 | 0.0064 |
| deepseek | Depaul-WH | rmse | FM | -0.0029 [-0.030, +0.026] | -0.3 | 0.8560 | 1.0000 |
| mistral | Tijuana-WH | mae | DeepFM | +0.0357 [+0.011, +0.062] | +7.2 | 0.0056 | 0.2520 |
| mistral | Tijuana-WH | rmse | DeepFM | -0.0026 [-0.031, +0.026] | -0.3 | 0.8738 | 1.0000 |
| eurollm | Tijuana-WH | mae | DeepFM | +0.0820 [+0.040, +0.124] | +16.5 | 0.0002 | 0.0104 |
| eurollm | Tijuana-WH | rmse | DeepFM | +0.0418 [-0.010, +0.094] | +5.1 | 0.1126 | 1.0000 |
| llama | Tijuana-WH | mae | DeepFM | +0.0595 [+0.033, +0.086] | +11.9 | 0.0001 | 0.0064 |
| llama | Tijuana-WH | rmse | DeepFM | +0.0170 [-0.011, +0.046] | +2.1 | 0.2360 | 1.0000 |
| deepseek | Tijuana-WH | mae | DeepFM | +0.0000 [+0.000, +0.000] | +0.0 | 1.0000 | 1.0000 |
| deepseek | Tijuana-WH | rmse | DeepFM | +0.0000 [+0.000, +0.000] | +0.0 | 1.0000 | 1.0000 |
| mistral | LDOS-H | mae | DeepFM | -0.0339 [-0.070, +0.003] | -4.5 | 0.0726 | 1.0000 |
| mistral | LDOS-H | rmse | DeepFM | -0.0271 [-0.068, +0.015] | -2.9 | 0.2062 | 1.0000 |
| eurollm | LDOS-H | mae | DeepFM | +0.0226 [-0.015, +0.059] | +3.0 | 0.2304 | 1.0000 |
| eurollm | LDOS-H | rmse | DeepFM | +0.0281 [-0.010, +0.066] | +3.0 | 0.1520 | 1.0000 |
| llama | LDOS-H | mae | DeepFM | +0.0350 [-0.003, +0.074] | +4.7 | 0.0750 | 1.0000 |
| llama | LDOS-H | rmse | DeepFM | +0.0291 [-0.012, +0.073] | +3.1 | 0.1670 | 1.0000 |
| deepseek | LDOS-H | mae | DeepFM | +0.0042 [-0.033, +0.042] | +0.6 | 0.8276 | 1.0000 |
| deepseek | LDOS-H | rmse | DeepFM | +0.0002 [-0.038, +0.040] | +0.0 | 0.9892 | 1.0000 |
| mistral | Depaul-H | mae | CAMF_CU | -0.0008 [-0.033, +0.032] | -0.1 | 0.9768 | 1.0000 |
| mistral | Depaul-H | rmse | FM | +0.0132 [-0.032, +0.060] | +1.3 | 0.5542 | 1.0000 |
| eurollm | Depaul-H | mae | CAMF_CU | +0.0365 [+0.010, +0.064] | +5.1 | 0.0068 | 0.2992 |
| eurollm | Depaul-H | rmse | FM | -0.0025 [-0.033, +0.028] | -0.3 | 0.8870 | 1.0000 |
| llama | Depaul-H | mae | CAMF_CU | +0.0129 [-0.014, +0.040] | +1.8 | 0.3538 | 1.0000 |
| llama | Depaul-H | rmse | FM | -0.0253 [-0.058, +0.008] | -2.6 | 0.1470 | 1.0000 |
| deepseek | Depaul-H | mae | CAMF_CU | +0.0083 [-0.015, +0.032] | +1.2 | 0.4816 | 1.0000 |
| deepseek | Depaul-H | rmse | FM | -0.0034 [-0.040, +0.034] | -0.3 | 0.8682 | 1.0000 |
| mistral | Restaurant-H | mae | DeepFM | +0.0250 [-0.006, +0.057] | +5.0 | 0.1142 | 1.0000 |
| mistral | Restaurant-H | rmse | DeepFM | +0.0163 [-0.022, +0.057] | +2.0 | 0.4230 | 1.0000 |
| eurollm | Restaurant-H | mae | DeepFM | +0.0076 [-0.001, +0.017] | +1.5 | 0.0896 | 1.0000 |
| eurollm | Restaurant-H | rmse | DeepFM | -0.0038 [-0.014, +0.007] | -0.5 | 0.4978 | 1.0000 |
| llama | Restaurant-H | mae | DeepFM | +0.0248 [+0.006, +0.043] | +5.0 | 0.0100 | 0.4300 |
| llama | Restaurant-H | rmse | DeepFM | -0.0023 [-0.023, +0.018] | -0.3 | 0.8530 | 1.0000 |
| deepseek | Restaurant-H | mae | DeepFM | +0.0332 [+0.012, +0.055] | +6.7 | 0.0028 | 0.1372 |
| deepseek | Restaurant-H | rmse | DeepFM | +0.0080 [-0.014, +0.032] | +1.0 | 0.4852 | 1.0000 |
| mistral | YELP-WH | mae | BiasedMF | +0.0065 [+0.001, +0.012] | +0.8 | 0.0202 | 0.8484 |
| mistral | YELP-WH | rmse | BiasedMF | -0.0029 [-0.009, +0.003] | -0.3 | 0.3422 | 1.0000 |
| eurollm | YELP-WH | mae | BiasedMF | +0.0079 [+0.003, +0.013] | +1.0 | 0.0032 | 0.1504 |
| eurollm | YELP-WH | rmse | BiasedMF | -0.0010 [-0.007, +0.005] | -0.1 | 0.7350 | 1.0000 |
| llama | YELP-WH | mae | BiasedMF | +0.0084 [+0.003, +0.014] | +1.0 | 0.0028 | 0.1372 |
| llama | YELP-WH | rmse | BiasedMF | -0.0017 [-0.008, +0.004] | -0.2 | 0.5832 | 1.0000 |
| deepseek | YELP-WH | mae | BiasedMF | +0.0078 [+0.002, +0.013] | +1.0 | 0.0040 | 0.1840 |
| deepseek | YELP-WH | rmse | BiasedMF | -0.0008 [-0.007, +0.005] | -0.1 | 0.7778 | 1.0000 |
| mistral | YELP-H | mae | BiasedMF | -0.0199 [-0.028, -0.011] | -2.5 | 0.0001 | 0.0064 |
| mistral | YELP-H | rmse | BiasedMF | -0.0252 [-0.035, -0.015] | -2.5 | 0.0001 | 0.0064 |
| eurollm | YELP-H | mae | BiasedMF | -0.0186 [-0.026, -0.011] | -2.3 | 0.0001 | 0.0064 |
| eurollm | YELP-H | rmse | BiasedMF | -0.0249 [-0.033, -0.016] | -2.4 | 0.0001 | 0.0064 |
| llama | YELP-H | mae | BiasedMF | -0.0162 [-0.024, -0.008] | -2.0 | 0.0002 | 0.0104 |
| llama | YELP-H | rmse | BiasedMF | -0.0226 [-0.032, -0.013] | -2.2 | 0.0001 | 0.0064 |
| deepseek | YELP-H | mae | BiasedMF | -0.0172 [-0.025, -0.010] | -2.1 | 0.0001 | 0.0064 |
| deepseek | YELP-H | rmse | BiasedMF | -0.0242 [-0.033, -0.015] | -2.4 | 0.0001 | 0.0064 |

## F2: 64 tests, significant before / after Holm: 56 / 48
- lost: deepseek LDOS-WH mae vs LLM alone (-0.054, p=0.0042)
- lost: llama LDOS-H mae vs LLM alone (-0.042, p=0.0074)
- lost: eurollm Depaul-H rmse vs LLM alone (-0.068, p=0.0160)
- lost: mistral Restaurant-H mae vs LLM alone (-0.075, p=0.0338)
- lost: eurollm Restaurant-H mae vs LLM alone (-0.107, p=0.0044)
- lost: mistral YELP-H rmse vs LLM alone (-0.016, p=0.0074)
- lost: eurollm YELP-H rmse vs LLM alone (-0.014, p=0.0380)
- lost: deepseek YELP-H rmse vs LLM alone (-0.015, p=0.0212)

| LLM | setting | metric | vs | diff [95% CI] | rel. % | p | p Holm |
|---|---|---|---|---|---|---|---|
| mistral | LDOS-WH | mae | LLM alone | -0.0078 [-0.061, +0.045] | -1.0 | 0.7586 | 1.0000 |
| mistral | LDOS-WH | rmse | LLM alone | +0.0318 [-0.025, +0.091] | +3.3 | 0.2858 | 1.0000 |
| eurollm | LDOS-WH | mae | LLM alone | -0.0570 [-0.094, -0.020] | -6.9 | 0.0028 | 0.0480 |
| eurollm | LDOS-WH | rmse | LLM alone | -0.0959 [-0.134, -0.056] | -8.6 | 0.0001 | 0.0064 |
| llama | LDOS-WH | mae | LLM alone | -0.0688 [-0.112, -0.028] | -8.3 | 0.0024 | 0.0480 |
| llama | LDOS-WH | rmse | LLM alone | -0.1565 [-0.204, -0.109] | -13.7 | 0.0001 | 0.0064 |
| deepseek | LDOS-WH | mae | LLM alone | -0.0544 [-0.091, -0.018] | -6.7 | 0.0042 | 0.0672 |
| deepseek | LDOS-WH | rmse | LLM alone | -0.1176 [-0.160, -0.076] | -10.7 | 0.0001 | 0.0064 |
| mistral | Depaul-WH | mae | LLM alone | -0.4761 [-0.540, -0.413] | -38.3 | 0.0001 | 0.0064 |
| mistral | Depaul-WH | rmse | LLM alone | -0.4511 [-0.521, -0.381] | -31.5 | 0.0001 | 0.0064 |
| eurollm | Depaul-WH | mae | LLM alone | -0.3912 [-0.452, -0.332] | -33.8 | 0.0001 | 0.0064 |
| eurollm | Depaul-WH | rmse | LLM alone | -0.3917 [-0.463, -0.323] | -28.5 | 0.0001 | 0.0064 |
| llama | Depaul-WH | mae | LLM alone | -0.3914 [-0.453, -0.332] | -33.8 | 0.0001 | 0.0064 |
| llama | Depaul-WH | rmse | LLM alone | -0.3991 [-0.471, -0.328] | -28.9 | 0.0001 | 0.0064 |
| deepseek | Depaul-WH | mae | LLM alone | -0.3735 [-0.432, -0.317] | -32.4 | 0.0001 | 0.0064 |
| deepseek | Depaul-WH | rmse | LLM alone | -0.3919 [-0.460, -0.325] | -28.5 | 0.0001 | 0.0064 |
| mistral | Tijuana-WH | mae | LLM alone | -0.2117 [-0.301, -0.126] | -28.4 | 0.0001 | 0.0064 |
| mistral | Tijuana-WH | rmse | LLM alone | -0.3716 [-0.491, -0.256] | -31.1 | 0.0001 | 0.0064 |
| eurollm | Tijuana-WH | mae | LLM alone | -0.2314 [-0.295, -0.172] | -28.5 | 0.0001 | 0.0064 |
| eurollm | Tijuana-WH | rmse | LLM alone | -0.2508 [-0.334, -0.172] | -22.5 | 0.0001 | 0.0064 |
| llama | Tijuana-WH | mae | LLM alone | -0.2890 [-0.367, -0.214] | -34.2 | 0.0001 | 0.0064 |
| llama | Tijuana-WH | rmse | LLM alone | -0.3733 [-0.467, -0.284] | -30.7 | 0.0001 | 0.0064 |
| deepseek | Tijuana-WH | mae | LLM alone | -0.2666 [-0.353, -0.184] | -34.9 | 0.0001 | 0.0064 |
| deepseek | Tijuana-WH | rmse | LLM alone | -0.2725 [-0.374, -0.177] | -24.8 | 0.0001 | 0.0064 |
| mistral | LDOS-H | mae | LLM alone | -0.1311 [-0.188, -0.076] | -15.6 | 0.0001 | 0.0064 |
| mistral | LDOS-H | rmse | LLM alone | -0.2518 [-0.314, -0.188] | -21.5 | 0.0001 | 0.0064 |
| eurollm | LDOS-H | mae | LLM alone | -0.0591 [-0.096, -0.023] | -7.1 | 0.0014 | 0.0294 |
| eurollm | LDOS-H | rmse | LLM alone | -0.0954 [-0.135, -0.056] | -8.9 | 0.0001 | 0.0064 |
| llama | LDOS-H | mae | LLM alone | -0.0417 [-0.074, -0.011] | -5.1 | 0.0074 | 0.1036 |
| llama | LDOS-H | rmse | LLM alone | -0.0777 [-0.112, -0.043] | -7.4 | 0.0001 | 0.0064 |
| deepseek | LDOS-H | mae | LLM alone | -0.0543 [-0.088, -0.022] | -6.8 | 0.0012 | 0.0264 |
| deepseek | LDOS-H | rmse | LLM alone | -0.0891 [-0.125, -0.055] | -8.6 | 0.0001 | 0.0064 |
| mistral | Depaul-H | mae | LLM alone | -0.0834 [-0.131, -0.037] | -10.5 | 0.0002 | 0.0064 |
| mistral | Depaul-H | rmse | LLM alone | -0.1908 [-0.253, -0.129] | -16.0 | 0.0001 | 0.0064 |
| eurollm | Depaul-H | mae | LLM alone | +0.0137 [-0.031, +0.057] | +1.9 | 0.5472 | 1.0000 |
| eurollm | Depaul-H | rmse | LLM alone | -0.0677 [-0.126, -0.012] | -6.4 | 0.0160 | 0.1920 |
| llama | Depaul-H | mae | LLM alone | -0.0147 [-0.058, +0.028] | -2.0 | 0.5120 | 1.0000 |
| llama | Depaul-H | rmse | LLM alone | -0.1402 [-0.206, -0.078] | -12.7 | 0.0001 | 0.0064 |
| deepseek | Depaul-H | mae | LLM alone | -0.0780 [-0.120, -0.037] | -9.8 | 0.0002 | 0.0064 |
| deepseek | Depaul-H | rmse | LLM alone | -0.1356 [-0.189, -0.083] | -12.1 | 0.0001 | 0.0064 |
| mistral | Restaurant-H | mae | LLM alone | -0.0753 [-0.146, -0.006] | -12.6 | 0.0338 | 0.3380 |
| mistral | Restaurant-H | rmse | LLM alone | -0.1954 [-0.306, -0.094] | -18.9 | 0.0002 | 0.0064 |
| eurollm | Restaurant-H | mae | LLM alone | -0.1067 [-0.182, -0.035] | -17.4 | 0.0044 | 0.0672 |
| eurollm | Restaurant-H | rmse | LLM alone | -0.1945 [-0.289, -0.103] | -19.2 | 0.0001 | 0.0064 |
| llama | Restaurant-H | mae | LLM alone | -0.1041 [-0.175, -0.038] | -16.6 | 0.0024 | 0.0480 |
| llama | Restaurant-H | rmse | LLM alone | -0.2236 [-0.316, -0.136] | -21.4 | 0.0001 | 0.0064 |
| deepseek | Restaurant-H | mae | LLM alone | -0.1444 [-0.215, -0.078] | -21.4 | 0.0001 | 0.0064 |
| deepseek | Restaurant-H | rmse | LLM alone | -0.2755 [-0.371, -0.185] | -24.9 | 0.0001 | 0.0064 |
| mistral | YELP-WH | mae | LLM alone | -0.0524 [-0.067, -0.038] | -6.0 | 0.0001 | 0.0064 |
| mistral | YELP-WH | rmse | LLM alone | -0.0594 [-0.078, -0.041] | -5.5 | 0.0001 | 0.0064 |
| eurollm | YELP-WH | mae | LLM alone | -0.0689 [-0.085, -0.053] | -7.8 | 0.0001 | 0.0064 |
| eurollm | YELP-WH | rmse | LLM alone | -0.0686 [-0.088, -0.050] | -6.3 | 0.0001 | 0.0064 |
| llama | YELP-WH | mae | LLM alone | -0.0575 [-0.072, -0.043] | -6.6 | 0.0001 | 0.0064 |
| llama | YELP-WH | rmse | LLM alone | -0.0648 [-0.083, -0.047] | -6.0 | 0.0001 | 0.0064 |
| deepseek | YELP-WH | mae | LLM alone | -0.0625 [-0.079, -0.046] | -7.1 | 0.0001 | 0.0064 |
| deepseek | YELP-WH | rmse | LLM alone | -0.0686 [-0.088, -0.050] | -6.3 | 0.0001 | 0.0064 |
| mistral | YELP-H | mae | LLM alone | +0.0020 [-0.008, +0.012] | +0.3 | 0.6976 | 1.0000 |
| mistral | YELP-H | rmse | LLM alone | -0.0160 [-0.028, -0.004] | -1.6 | 0.0074 | 0.1036 |
| eurollm | YELP-H | mae | LLM alone | +0.0038 [-0.007, +0.015] | +0.5 | 0.4986 | 1.0000 |
| eurollm | YELP-H | rmse | LLM alone | -0.0137 [-0.027, -0.001] | -1.4 | 0.0380 | 0.3420 |
| llama | YELP-H | mae | LLM alone | -0.0017 [-0.012, +0.008] | -0.2 | 0.7304 | 1.0000 |
| llama | YELP-H | rmse | LLM alone | -0.0172 [-0.029, -0.006] | -1.7 | 0.0024 | 0.0480 |
| deepseek | YELP-H | mae | LLM alone | +0.0021 [-0.009, +0.013] | +0.3 | 0.7138 | 1.0000 |
| deepseek | YELP-H | rmse | LLM alone | -0.0154 [-0.029, -0.002] | -1.5 | 0.0212 | 0.2332 |

## F3: 24 tests, significant before / after Holm: 18 / 13
- lost: eurollm MovieLens auc vs BiasedMF (+0.020, p=0.0280)
- lost: llama MovieLens auc vs BiasedMF (+0.024, p=0.0070)
- lost: deepseek MovieLens auc vs BiasedMF (+0.021, p=0.0210)
- lost: mistral AmazonBooks auc vs DeepFM (+0.038, p=0.0160)
- lost: llama AmazonBooks auc vs DeepFM (+0.029, p=0.0340)

| LLM | setting | metric | vs | diff [95% CI] | rel. % | p | p Holm |
|---|---|---|---|---|---|---|---|
| mistral | MovieLens | rmse | BiasedMF | -0.0354 [-0.054, -0.017] | -3.9 | 0.0001 | 0.0024 |
| mistral | MovieLens | mae | BiasedMF | -0.0250 [-0.041, -0.009] | -3.5 | 0.0024 | 0.0340 |
| mistral | MovieLens | auc | BiasedMF | +0.0284 [+0.012, +0.046] | +3.5 | 0.0030 | 0.0360 |
| eurollm | MovieLens | rmse | BiasedMF | -0.0305 [-0.049, -0.012] | -3.4 | 0.0004 | 0.0076 |
| eurollm | MovieLens | mae | BiasedMF | -0.0257 [-0.042, -0.009] | -3.6 | 0.0022 | 0.0340 |
| eurollm | MovieLens | auc | BiasedMF | +0.0202 [+0.002, +0.038] | +2.5 | 0.0280 | 0.2240 |
| llama | MovieLens | rmse | BiasedMF | -0.0320 [-0.051, -0.014] | -3.5 | 0.0004 | 0.0076 |
| llama | MovieLens | mae | BiasedMF | -0.0252 [-0.042, -0.009] | -3.5 | 0.0022 | 0.0340 |
| llama | MovieLens | auc | BiasedMF | +0.0243 [+0.007, +0.043] | +3.0 | 0.0070 | 0.0770 |
| deepseek | MovieLens | rmse | BiasedMF | -0.0302 [-0.049, -0.012] | -3.3 | 0.0020 | 0.0340 |
| deepseek | MovieLens | mae | BiasedMF | -0.0258 [-0.042, -0.009] | -3.6 | 0.0020 | 0.0340 |
| deepseek | MovieLens | auc | BiasedMF | +0.0214 [+0.005, +0.039] | +2.7 | 0.0210 | 0.1890 |
| mistral | AmazonBooks | rmse | MLP | -0.0339 [-0.047, -0.021] | -4.2 | 0.0001 | 0.0024 |
| mistral | AmazonBooks | mae | MLP | -0.0057 [-0.015, +0.003] | -1.0 | 0.2236 | 1.0000 |
| mistral | AmazonBooks | auc | DeepFM | +0.0375 [+0.006, +0.071] | +5.0 | 0.0160 | 0.1600 |
| eurollm | AmazonBooks | rmse | MLP | -0.0232 [-0.033, -0.014] | -2.9 | 0.0001 | 0.0024 |
| eurollm | AmazonBooks | mae | MLP | +0.0045 [-0.003, +0.012] | +0.8 | 0.2486 | 1.0000 |
| eurollm | AmazonBooks | auc | DeepFM | +0.0143 [-0.011, +0.041] | +1.9 | 0.2930 | 1.0000 |
| llama | AmazonBooks | rmse | MLP | -0.0301 [-0.042, -0.018] | -3.7 | 0.0001 | 0.0024 |
| llama | AmazonBooks | mae | MLP | -0.0020 [-0.011, +0.007] | -0.4 | 0.6556 | 1.0000 |
| llama | AmazonBooks | auc | DeepFM | +0.0291 [+0.002, +0.058] | +3.9 | 0.0340 | 0.2380 |
| deepseek | AmazonBooks | rmse | MLP | -0.0271 [-0.038, -0.016] | -3.4 | 0.0001 | 0.0024 |
| deepseek | AmazonBooks | mae | MLP | +0.0013 [-0.007, +0.010] | +0.2 | 0.7758 | 1.0000 |
| deepseek | AmazonBooks | auc | DeepFM | +0.0196 [-0.006, +0.047] | +2.6 | 0.1520 | 0.9120 |
