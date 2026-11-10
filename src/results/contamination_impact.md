# Contamination probes and their effect on the results

## Probe results (pretrained base models)

| LLM | ML items exact | ML items no year | ML rating exact (mode ctrl) | ML timestamp exact (prev-line ctrl) | AZ items exact | AZ items fuzzy |
|---|---|---|---|---|---|---|
| mistral | 3.7% | 3.8% | 33.6% (36.0%) | 0.8% (0.8%) | 0.0% | 0.0% |
| eurollm | 1.6% | 1.8% | 33.7% (36.0%) | 0.8% (0.8%) | 0.0% | 0.0% |
| llama | 2.5% | 2.7% | 32.4% (36.0%) | 0.6% (0.8%) | 0.0% | 0.0% |
| deepseek | 2.3% | 2.3% | 34.8% (36.0%) | 0.8% (0.8%) | 0.0% | 0.0% |

## MovieLens: test interactions split by memorization

| LLM | group | n | LLM alone RMSE / MAE | HyCPR-LLM RMSE / MAE | BiasedMF RMSE / MAE | HyCPR − BiasedMF: RMSE | MAE |
|---|---|---|---|---|---|---|---|
| mistral | item memorized: yes | 68 | 0.782 / 0.631 | 0.775 / 0.626 | 0.906 / 0.719 | -0.131 [-0.261, -0.011] * | -0.092 [-0.188, +0.002] |
| mistral | item memorized: no | 1932 | 0.886 / 0.699 | 0.872 / 0.689 | 0.905 / 0.711 | -0.032 [-0.052, -0.014] * | -0.023 [-0.040, -0.006] * |
| mistral | rating line memorized: yes | 4 | (too few) | | | | |
| mistral | rating line memorized: no | 1996 | 0.884 / 0.697 | 0.870 / 0.687 | 0.905 / 0.713 | -0.035 [-0.054, -0.017] * | -0.025 [-0.041, -0.009] * |
| eurollm | item memorized: yes | 31 | 0.717 / 0.579 | 0.708 / 0.569 | 0.772 / 0.629 | -0.063 [-0.218, +0.078] | -0.061 [-0.173, +0.044] |
| eurollm | item memorized: no | 1969 | 0.891 / 0.697 | 0.877 / 0.688 | 0.907 / 0.713 | -0.030 [-0.049, -0.012] * | -0.025 [-0.042, -0.009] * |
| eurollm | rating line memorized: yes | 7 | (too few) | | | | |
| eurollm | rating line memorized: no | 1993 | 0.890 / 0.696 | 0.875 / 0.687 | 0.906 / 0.713 | -0.031 [-0.049, -0.012] * | -0.026 [-0.042, -0.009] * |
| llama | item memorized: yes | 47 | 0.906 / 0.698 | 0.907 / 0.702 | 1.026 / 0.804 | -0.119 [-0.260, +0.000] | -0.102 [-0.211, +0.002] |
| llama | item memorized: no | 1953 | 0.887 / 0.696 | 0.872 / 0.686 | 0.902 / 0.709 | -0.030 [-0.049, -0.011] * | -0.023 [-0.040, -0.007] * |
| llama | rating line memorized: yes | 2 | (too few) | | | | |
| llama | rating line memorized: no | 1998 | 0.887 / 0.696 | 0.873 / 0.687 | 0.905 / 0.712 | -0.032 [-0.051, -0.013] * | -0.025 [-0.041, -0.009] * |
| deepseek | item memorized: yes | 41 | 0.842 / 0.665 | 0.831 / 0.668 | 0.917 / 0.732 | -0.086 [-0.258, +0.068] | -0.064 [-0.191, +0.065] |
| deepseek | item memorized: no | 1959 | 0.890 / 0.697 | 0.875 / 0.686 | 0.904 / 0.711 | -0.029 [-0.048, -0.011] * | -0.025 [-0.042, -0.008] * |
| deepseek | rating line memorized: yes | 3 | (too few) | | | | |
| deepseek | rating line memorized: no | 1997 | 0.890 / 0.697 | 0.875 / 0.687 | 0.905 / 0.712 | -0.030 [-0.049, -0.012] * | -0.026 [-0.042, -0.009] * |

## AmazonBooks: test interactions split by memorization

| LLM | group | n | LLM alone RMSE / MAE | HyCPR-LLM RMSE / MAE | MLP RMSE / MAE | HyCPR − MLP: RMSE | MAE |
|---|---|---|---|---|---|---|---|
| mistral | item memorized: yes | 0 | (too few) | | | | |
| mistral | item memorized: no | 2000 | 0.780 / 0.555 | 0.771 / 0.545 | 0.805 / 0.551 | -0.034 [-0.047, -0.021] * | -0.006 [-0.015, +0.003] |
| eurollm | item memorized: yes | 0 | (too few) | | | | |
| eurollm | item memorized: no | 2000 | 0.798 / 0.578 | 0.781 / 0.555 | 0.805 / 0.551 | -0.023 [-0.033, -0.014] * | +0.004 [-0.003, +0.012] |
| llama | item memorized: yes | 0 | (too few) | | | | |
| llama | item memorized: no | 2000 | 0.786 / 0.563 | 0.774 / 0.549 | 0.805 / 0.551 | -0.030 [-0.042, -0.018] * | -0.002 [-0.011, +0.007] |
| deepseek | item memorized: yes | 0 | (too few) | | | | |
| deepseek | item memorized: no | 2000 | 0.791 / 0.569 | 0.777 / 0.552 | 0.805 / 0.551 | -0.027 [-0.038, -0.016] * | +0.001 [-0.007, +0.010] |
