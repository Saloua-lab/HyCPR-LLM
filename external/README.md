# Official code of the LLM–CF baselines

`src/llara_rating.py` and `src/allmrec_rating.py` import modules from the official repositories of LLaRA and
A-LLMRec. `src/collm.py` follows the settings of CoLLM. Clone the repositories
here, at the commits we used:

```bash
cd external
git clone https://github.com/ljy0ustc/LLaRA.git    && git -C LLaRA    checkout 9733f732399e95e236f4f2e1d1139c1f0437090e
git clone https://github.com/ghdtjr/A-LLMRec.git   && git -C A-LLMRec checkout e216bbc3d0c9eec548f2d9667bdeca8019f29af1
git clone https://github.com/zyang1580/CoLLM.git   && git -C CoLLM    checkout df1a24dcc083ce9568565b72def58c7af80cd8c9
```

Used modules:

- LLaRA: `model/mlp_projector.MlpProjector`, `recommender/A_SASRec_final_bce_llm.SASRec`.
- A-LLMRec: `models/a_llmrec_model.two_layer_mlp`, `pre_train/sasrec/model.SASRec`.
  Importing them needs `pytz`; `sentence_transformers` is replaced by a stub in our script.

Each repository keeps its own licence.
