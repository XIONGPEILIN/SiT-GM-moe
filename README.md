## Markov Superposition for Joint Continuous-Discrete Generative Modeling (Jump+Flow)
<br><sub>Based on SiT (Scalable Interpolant Transformers)</sub>

### [Paper (arXiv:2410.20587v3)](https://arxiv.org/abs/2410.20587) 

This repository contains PyTorch model definitions, pre-trained weights, and training/sampling code for exploring the **Jump+Flow** framework based on the paper **Markov Superposition for Joint Continuous-Discrete Generative Modeling**. 

This codebase builds upon the Scalable Interpolant Transformers (SiT) architecture to jointly model continuous flows and discrete jumps, effectively resolving the Jump+Flow generation framework.

> [**Markov Superposition for Joint Continuous-Discrete Generative Modeling**](https://arxiv.org/abs/2410.20587)<br>
> arXiv:2410.20587v3

The Jump+Flow model enables generative pathways where states can either evolve continuously (Flow) or undergo instantaneous transitions (Jump) directly to targeted latent representations.

### What is Jump+Flow?
By introducing an additional *jump* mechanism to standard flow-matching/diffusion ODEs, the generative process is modeled as a Markov superposition of a continuous drift and a jump process. 
- **Flow head**: predicts the continuous velocity ($v_t$).
- **Jump head**: predicts the target landing position ($jump\_d\_\theta$) and the jump intensity/rate ($\lambda_t$).
During sampling, paths probabilistically switch from continuous evolution to discrete jumps directly towards the target data distribution.

## Setup

First, clone and set up the repository:

```bash
git clone https://github.com/willisma/SiT.git
cd SiT
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment:

```bash
conda env create -f environment.yml
conda activate SiT
```

## Sampling (Jump+Flow)

You can sample from trained models with [`sample.py`](sample.py) using the `JUMP_FLOW` mode. 

To sample using the Jump+Flow stochastic Euler sampler:
```bash
python sample.py JUMP_FLOW \
  --model SiT-XL/2 \
  --image-size 256 \
  --ckpt /path/to/your/checkpoint.safetensors \
  --stochastic-jump
```

You can view step-by-step jump probability maps by adapting the sampler to track `p_jump`. The jump probabilities organically scale to `1.0` at the end of the trajectory, forcing all remaining tokens to jump to their terminal data states.

## Training Jump+Flow Models

We provide a training script for SiT with Jump+Flow heads in [`train.py`](train.py). To launch training on `N` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py \
  --model SiT-XL/2 \
  --data-path /path/to/imagenet/train
```

## License
This project is under the MIT license. See [LICENSE](LICENSE.txt) for details.
