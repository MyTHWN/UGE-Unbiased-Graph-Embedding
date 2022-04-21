
# UGE-Unbiased-Graph-Embedding

This is the code for paper "Unbiased Graph Embedding with Biased Graph Observations" accepted by WWW'22.

## Requirements

Code is tested in Python 3.8 and PyTorch 1.8~1.11. Some major requirements are listed below:

```
torch
dgl
pandas
numpy
scipy
tqdm
pickle
sklearn
```

## Datasets

<p>We evaluate our method on three datasets: Pokec-z, Pokec-n and Movielens-1M. Raw data are uploaded or will be automatically downloaded to <code>./raw_data</code> folder, and they will be further converted to a unified csv format in <code>./processed_data</code> folder.</p>

1. **Pokec-z** and **Pokec-n** are sampled from [soc_Pokec](http://snap.stanford.edu/data/soc-Pokec.html) following [FairGNN](https://github.com/EnyanDai/FairGNN). Raw and processed data have been already uploaded to <code>./raw_data</code> and <code>./processed_data</code> folders. 
2. [**Movielens-1M**](https://grouplens.org/datasets/movielens/1m/) is too large to hold on github, so it will be automatically downloaded to <code>./raw_data</code> folder and be processed to <code>./processed_data</code> folder when launching UGE training process. 


## UGE Training

## Cite

Please cite our paper if you are inspired by our work.
> @article{wang2021unbiased,
>  title={Unbiased Graph Embedding with Biased Graph Observations},
>  author={Wang, Nan and Lin, Lu and Li, Jundong and Wang, Hongning},
>  journal={arXiv preprint arXiv:2110.13957},
>  year={2021}
}


