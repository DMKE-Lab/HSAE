# Hierarchical Self-Attention Embedding for Temporal Knowledge Graph Completion

### Installation
```
conda create -n hsae python=3.7

conda activate hsae

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1

conda install numpy==1.21.2
```


## How to run

```
python main.py -neg_ratio 500-se_prop 0.68 -model HSAE_simple --dataset icews14
```

