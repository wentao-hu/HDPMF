# Decentralized MF with HDP 

This is our implementation for [dencentralized matrix factorization with heterogeneous differential privacy](https://arxiv.org/pdf/2212.00306.pdf):


## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Run Our method HDPMF :
```
python mf_hdp_centralized.py --data Data/ml-1m --max_budget 1 --nonprivate_epochs 54 --private_epochs 54 --embedding_dim 8 --regularization 0.005 --learning_rate 0.002 --stddev 0.1

```

Run baseline method PDPMF:
```
python mf_sampling_centralized.py --data Data/ml-1m --max_budget 1 --threshold 1 --nonprivate_epochs 54 --private_epochs 54 --embedding_dim 8 --regularization 0.005 --learning_rate 0.002 --stddev 0.1
```

## Code Reference
[Neural Collaborative Filtering vs. Matrix Factorization Revisited](https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity)
