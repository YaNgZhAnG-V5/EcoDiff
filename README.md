# EcoDiff

## Introduction
EcoDiff is a diffusion model pruning method that can prune the model with minimal performance loss. We show that we can prune 20% of the model parameters while keeping the performance close to the original model **without retraining or healing the model after pruning**.

## Run inference on EcoDiff Pruned Model
### create environment
```bash
conda create -n ecodiff python=3.10
```

### install lib
```bash
pip install -r requirements.txt
```

### run inference on eco-diff
```bash
python inference.py
```

## Cite our work
```
@misc{zhang2024effortlessefficiencylowcostpruning,
      title={Effortless Efficiency: Low-Cost Pruning of Diffusion Models}, 
      author={Yang Zhang and Er Jin and Yanfei Dong and Ashkan Khakzar and Philip Torr and Johannes Stegmaier and Kenji Kawaguchi},
      year={2024},
      eprint={2412.02852},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.02852}, 
}
```