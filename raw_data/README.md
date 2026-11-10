# Public datasets

Download the datasets into this folder. The scripts read them from these paths:

| Dataset | Files | Source |
|---|---|---|
| MovieLens-1M | `movielens/ml-1m/{ratings,movies,users}.dat` | https://grouplens.org/datasets/movielens/1m/ |
| Amazon-Books (2018, 5-core) | `amazon/Books_5.json.gz`, `amazon/meta_Books.json.gz` | https://nijianmo.github.io/amazon/ |
| Yelp Open Dataset | `yelp/yelp_academic_dataset_{review,business,user}.json` | https://www.yelp.com/dataset |
| MIND | `mind/MINDsmall_train.zip`, `mind/MINDsmall_dev.zip` | https://msnews.github.io/ |

Then build the prompt datasets: `cd src && python prepare_large.py {movielens,amazon,yelp,mind}`.
