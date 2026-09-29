# Fourier Learning Machines (FLMs)
This repo contains the PyTorch implementation of the FLM architecture from the paper [Fourier Learning Machines: Nonharmonic Fourier-Based Neural Networks for Scientific Machine Learning](https://openreview.net/forum?id=LPKt5vd7yz) (TMLR, 2025).

## Files

- `flm.py`: the FLM code. Download this file and import it into your own code.
- `pytorch_FLM_demo.ipynb`: a demo notebook that explains the architecture and trains an FLM on a 2D example.
- `requirements.txt`: the Python packages needed.

## Installation

```bash
pip install -r requirements.txt
```

`flm.py` itself only needs PyTorch. NumPy and Matplotlib are used by the demo notebook.

## Usage

Put `flm.py` in the same folder as your code, then:

```python
import torch
from flm import FourierLearningMachine, to_reference_domain

model = FourierLearningMachine(m=2, N=16, seed=0)   # m = input dimension, N = number of sub-networks

x = torch.zeros(2)          # one point in [-pi, pi]^2
y = model(x)                # scalar output

X = torch.rand(100, 2) * 2 * torch.pi - torch.pi    # a batch of points
Y = torch.vmap(model)(X)    # shape (100,)
```

The model expects inputs in $[-\pi, \pi]^m$. If your data lives on another box, map it first with `to_reference_domain(x, lows, highs)`.

`flm.py` also provides `lexi_sign_matrix(m)` (the fixed m-Lexi Sign Matrix) and `lattice_frequencies(N, m)` (the default frequency initialization). The model uses both internally, but you can call them directly, for example to try a different initialization. See `pytorch_FLM_demo.ipynb` for a full training example.

## Citation

To cite the published paper, please use

```bibtex
@article{rubel2025fourier,
  title={Fourier Learning Machines: Nonharmonic Fourier-Based Neural Networks for Scientific Machine Learning},
  author={Mominul Rubel and Adam Meyers and Gabriel Nicolosi},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=LPKt5vd7yz}
}
```
