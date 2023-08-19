# HDPMF

This is our implementation for [dencentralized matrix factorization with heterogeneous differential privacy](https://arxiv.org/pdf/2212.00306.pdf):


## How to run
1. Download datasets
- [ML-100K/ML-1M](https://grouplens.org/datasets/movielens/)
- [Douban](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information)
- [Yelp](https://www.yelp.com/dataset)

2. Preprocess datasets and put them in the folder ```/Data```
- For default setting or changing hyperparmeters experiments
```
python dataprocess_for_default.py 
```
- For changing dataset sparsity experiments
```
python dataprocess_for_sparsity.py 
```

3. Run model

- Run Our model HDPMF :
```
python mf_hdp.py --data Data/ml-1m  --lr 0.01 --embedding_dim 10 --regularization 0.01  --stddev 0.1
```

- Run baseline method PDPMF:
```
python mf_sampling.py --data Data/ml-1m  --lr 0.01 --embedding_dim 10 --regularization 0.01  --stddev 0.1
```

- Run original MF without any noise:
```
python mf_nonprivate.py --data Data/ml-1m  --lr 0.01 --embedding_dim 10 --regularization 0.01 --stddev 0.1
```

## Requirements
```
python: 3.8
sklearn: 1.0.2
```

## Code Reference

- [Neural Collaborative Filtering vs. Matrix Factorization Revisited](https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity)
