# HyCPR-LLM (Mistral) minus best baseline, MAE, by segment (paired bootstrap 95% CI; * excludes 0)

## LDOS-H: vs DeepFM; fusion with CAMF_CUCI, w = 0.6; all: -0.034 [-0.070, +0.003]
- user activity [cuts 48/93 training ratings]: low (n=162): +0.002; mid (n=163): -0.026; high (n=127): -0.089*
- item popularity [cuts 1/2 training ratings]: low (n=259): -0.046; mid (n=92): -0.057; high (n=101): +0.017
- true rating: 1-2 (n=42): -0.078; 3 (n=108): +0.023; 4-5 (n=302): -0.048*

## Depaul-H: vs CAMF_CU; fusion with CAMF_CU, w = 0.55; all: -0.001 [-0.033, +0.032]
- user activity [cuts 29/49 training ratings]: low (n=228): +0.003; mid (n=244): +0.023; high (n=212): -0.032
- item popularity [cuts 41/53 training ratings]: low (n=249): +0.002; mid (n=244): -0.002; high (n=191): -0.003
- true rating: 1-2 (n=220): +0.045; 3 (n=140): -0.004; 4-5 (n=324): -0.030

## Restaurant-H: vs DeepFM; fusion with DeepFM, w = 0.4; all: +0.025 [-0.006, +0.057]
- user activity [cuts 28/48 training ratings]: low (n=85): -0.026; mid (n=84): -0.032; high (n=81): +0.137*
- item popularity [cuts 19/42 training ratings]: low (n=88): +0.111*; mid (n=98): -0.003; high (n=64): -0.050
- true rating: 1-2 (n=49): +0.111*; 3 (n=24): -0.066; 4-5 (n=177): +0.013

## YELP-H: vs BiasedMF; fusion with DeepFM, w = 0.45; all: -0.020 [-0.028, -0.011]
- user activity [cuts 16/45 training ratings]: low (n=689): -0.022*; mid (n=651): -0.006; high (n=660): -0.031*
- item popularity [cuts 34/92 training ratings]: low (n=674): -0.034*; mid (n=669): -0.015*; high (n=657): -0.010
- true rating: 1-2 (n=189): -0.045*; 3 (n=227): +0.011; 4-5 (n=1584): -0.021*

## MovieLens-H: vs BiasedMF; fusion with BiasedMF, w = 0.85; all: -0.025 [-0.041, -0.009]
- user activity [cuts 87/256 training ratings]: low (n=670): -0.062*; mid (n=672): -0.010; high (n=658): -0.003
- item popularity [cuts 298/717 training ratings]: low (n=669): -0.024; mid (n=664): -0.029*; high (n=667): -0.022
- true rating: 1-2 (n=352): -0.112*; 3 (n=576): -0.037*; 4-5 (n=1072): +0.010

## AmazonBooks-H: vs MLP; fusion with MLP, w = 0.6; all: -0.006 [-0.015, +0.003]
- user activity [cuts 6/28 training ratings]: low (n=695): +0.015; mid (n=647): -0.009; high (n=658): -0.024*
- item popularity [cuts 3/107 training ratings]: low (n=671): -0.020*; mid (n=665): -0.001; high (n=664): +0.004
- true rating: 1-2 (n=98): -0.095*; 3 (n=131): -0.073*; 4-5 (n=1771): +0.004
