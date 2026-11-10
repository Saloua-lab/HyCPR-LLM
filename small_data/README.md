# Small context-aware datasets

LDOS-CoMoDa, DePaul Movies and Tijuana Restaurants. The scripts expect, in this folder:

- the prompt datasets (Hugging Face `DatasetDict`s saved with `save_to_disk`, splits `train` / `test`, column
  `example`):
  - `LocationLDOSWithoutHis`, `LocationLDOSWithH`, `LocationLDOSWithHisAvecMistral`;
  - `Datasets/DepaulNoHistory`, `LocationDepaulAvecHis`, `LocationDepaulWithHisAvecMistral`,
    `LocationDepaulWithHisAvecLLama`;
  - `LocationTRestaurantWithoutHis`, `LocationRestaurantAvecHis`, `LocationRestaurantSansHis`,
    `LocationRestaurantWithHisAvecMistral`, `LocationRestaurantWithHisAvecLLama`;
  - the Llama variants of these sets;
- the structured sources of these prompts, which the CF baselines are trained on:
  - `Datasets/LDOSInitial/{train,test}_data.xlsx`;
  - `Datasets/DepaulMoviexlsx/Depaul_Movies_{Train,Test}.xlsx`;
  - `Datasets/TijuanaWithAlpaca/{train,test}_data.xlsx`.

`-H` sets include the user's history in the prompt, `-WH` sets do not. The jobs in `src/jobs/` that evaluate the
first-version -WH models also read those LoRA adapters from this folder (for example `LDOS-modelSansHist`).

The original datasets are cited in Section "Datasets" of the paper.
