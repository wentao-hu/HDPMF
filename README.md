# HDPMF

This is our implementation for [dencentralized matrix factorization with heterogeneous differential privacy](https://arxiv.org/pdf/2212.00306.pdf):


## Example to run the codes.

Run Our method HDPMF :
```
python mf_hdp.py --data Data/ml-1m  --lr 0.01 --embedding_dim 10 --regularization 0.01  --stddev 0.1

```

Run baseline method PDPMF:
```
python mf_sampling.py --data Data/ml-1m  --lr 0.01 --embedding_dim 10 --regularization 0.01  --stddev 0.1
```

Run nonprivate MF:
```
python mf_nonprivate.py --data Data/ml-1m  --lr 0.01 --embedding_dim 10 --regularization 0.01 --stddev 0.1
```

## Code Reference
[Neural Collaborative Filtering vs. Matrix Factorization Revisited](https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity)
