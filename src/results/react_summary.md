# ReAct with tools (Mistral-7B-Instruct, zero-shot)

## LDOS (452 test interactions)

| config | usable | MAE [CI] | RMSE [CI] | vs react_all: MAE | RMSE | tools/episode |
|---|---|---|---|---|---|---|
| direct | 452/452 | 0.805 [0.741, 0.872] | 1.066 [0.997, 1.138] | +0.033 [-0.033, +0.101] | +0.028 [-0.051, +0.109] |  |
| direct_tools | 401/452 | 0.884 [0.821, 0.948] | 1.119 [1.058, 1.179] | +0.112 [+0.043, +0.181] * | +0.081 [+0.009, +0.152] * |  |
| react_notools | 423/452 | 1.005 [0.934, 1.076] | 1.262 [1.188, 1.336] | +0.233 [+0.154, +0.313] * | +0.223 [+0.133, +0.315] * |  |
| react_all | 383/452 | 0.772 [0.709, 0.837] | 1.038 [0.959, 1.117] |  |  | user_ratings 0.70, item_ratings 0.51, similar_items 0.75, context_ratings 0.64, invented 0.27; episodes with an invented tool 22% |
| react_minus_user | 340/452 | 0.787 [0.729, 0.846] | 1.014 [0.942, 1.086] | +0.015 [-0.040, +0.068] | -0.025 [-0.090, +0.041] | user_ratings 0.04, item_ratings 0.85, similar_items 0.87, context_ratings 0.86, invented 0.34; episodes with an invented tool 23% |
| react_minus_item | 261/452 | 0.776 [0.718, 0.835] | 1.002 [0.932, 1.073] | +0.004 [-0.045, +0.053] | -0.036 [-0.099, +0.025] | user_ratings 1.01, similar_items 0.97, context_ratings 0.95, invented 0.06; episodes with an invented tool 4% |
| react_minus_similar | 211/452 | 0.779 [0.720, 0.839] | 1.012 [0.944, 1.081] | +0.006 [-0.047, +0.060] | -0.026 [-0.094, +0.039] | user_ratings 1.02, item_ratings 0.85, similar_items 0.00, context_ratings 0.84, invented 0.57; episodes with an invented tool 34% |
| react_minus_context | 336/452 | 0.774 [0.713, 0.837] | 1.025 [0.951, 1.100] | +0.002 [-0.049, +0.052] | -0.014 [-0.068, +0.041] | user_ratings 0.94, item_ratings 0.73, similar_items 0.79, context_ratings 0.01, invented 0.39; episodes with an invented tool 26% |

## Depaul (684 test interactions)

| config | usable | MAE [CI] | RMSE [CI] | vs react_all: MAE | RMSE | tools/episode |
|---|---|---|---|---|---|---|
| direct | 476/684 | 1.301 [1.227, 1.376] | 1.633 [1.557, 1.706] | +0.227 [+0.142, +0.312] * | +0.234 [+0.145, +0.319] * |  |
| direct_tools | 562/684 | 1.087 [1.023, 1.149] | 1.376 [1.316, 1.434] | +0.012 [-0.063, +0.086] | -0.024 [-0.103, +0.054] |  |
| react_notools | 602/684 | 1.405 [1.317, 1.494] | 1.831 [1.741, 1.919] | +0.331 [+0.230, +0.430] * | +0.431 [+0.331, +0.529] * |  |
| react_all | 677/684 | 1.074 [1.008, 1.143] | 1.400 [1.328, 1.472] |  |  | user_ratings 0.95, item_ratings 0.87, similar_items 0.92, context_ratings 0.85, invented 0.06; episodes with an invented tool 5% |
| react_minus_user | 636/684 | 1.095 [1.033, 1.160] | 1.388 [1.318, 1.459] | +0.021 [-0.045, +0.086] | -0.012 [-0.082, +0.057] | user_ratings 0.01, item_ratings 0.79, similar_items 0.76, context_ratings 0.86, invented 0.09; episodes with an invented tool 6% |
| react_minus_item | 678/684 | 1.058 [0.996, 1.121] | 1.354 [1.286, 1.424] | -0.016 [-0.083, +0.048] | -0.046 [-0.115, +0.022] | user_ratings 0.99, similar_items 0.96, context_ratings 0.98, invented 0.02; episodes with an invented tool 2% |
| react_minus_similar | 628/684 | 1.096 [1.033, 1.161] | 1.399 [1.330, 1.468] | +0.021 [-0.047, +0.088] | -0.001 [-0.074, +0.071] | user_ratings 1.08, item_ratings 0.98, context_ratings 0.98, invented 0.00; episodes with an invented tool 0% |
| react_minus_context | 669/684 | 1.178 [1.105, 1.251] | 1.527 [1.448, 1.603] | +0.104 [+0.026, +0.182] * | +0.127 [+0.047, +0.209] * | user_ratings 0.88, item_ratings 0.89, similar_items 0.85, invented 0.07; episodes with an invented tool 6% |

## Tijuana (250 test interactions)

| config | usable | MAE [CI] | RMSE [CI] | vs react_all: MAE | RMSE | tools/episode |
|---|---|---|---|---|---|---|
| direct | 171/250 | 1.245 [1.140, 1.352] | 1.512 [1.398, 1.623] | +0.162 [+0.047, +0.274] * | +0.089 [-0.031, +0.207] |  |
| direct_tools | 246/250 | 0.948 [0.844, 1.054] | 1.269 [1.165, 1.369] | -0.135 [-0.249, -0.020] * | -0.154 [-0.273, -0.033] * |  |
| react_notools | 200/250 | 1.338 [1.232, 1.446] | 1.605 [1.494, 1.717] | +0.254 [+0.128, +0.380] * | +0.182 [+0.050, +0.314] * |  |
| react_all | 221/250 | 1.083 [0.970, 1.198] | 1.423 [1.304, 1.540] |  |  | user_ratings 0.94, item_ratings 0.78, similar_items 0.77, context_ratings 0.87; episodes with an invented tool 0% |
| react_minus_user | 202/250 | 1.169 [1.050, 1.291] | 1.513 [1.382, 1.641] | +0.085 [-0.028, +0.201] | +0.091 [-0.029, +0.210] | user_ratings 0.00, item_ratings 0.88, similar_items 0.88, context_ratings 0.94, invented 0.04; episodes with an invented tool 2% |
| react_minus_item | 200/250 | 0.948 [0.831, 1.068] | 1.344 [1.214, 1.470] | -0.136 [-0.246, -0.026] * | -0.079 [-0.200, +0.037] | user_ratings 0.96, similar_items 0.93, context_ratings 0.99, invented 0.02; episodes with an invented tool 2% |
| react_minus_similar | 202/250 | 1.167 [1.036, 1.299] | 1.570 [1.428, 1.708] | +0.083 [-0.036, +0.203] | +0.148 [+0.013, +0.278] * | user_ratings 1.13, item_ratings 0.95, context_ratings 0.90, invented 0.01; episodes with an invented tool 1% |
| react_minus_context | 217/250 | 1.212 [1.097, 1.331] | 1.536 [1.410, 1.659] | +0.129 [+0.008, +0.251] * | +0.113 [-0.012, +0.239] | user_ratings 0.96, item_ratings 0.93, similar_items 0.83, invented 0.00; episodes with an invented tool 0% |
