# Equivariant Cellular Sheaves for Molecular Electronic Structure

[![DOI](https://zenodo.org/badge/1276323947.svg)](https://doi.org/10.5281/zenodo.20788374)

Code and paper source for **"Equivariant Cellular Sheaves for Molecular Electronic
Structure: Bridging Sheaf Cohomology and *E*(3)-Equivariant Hamiltonian Learning"**
(Krishna Harish).

The central result is that, in a localized atomic-orbital or Wannier basis, the
molecular single-particle Hamiltonian, after a constant positive-semidefinite
energy shift, is exactly the Laplacian of an *O*(3)-equivariant cellular sheaf on
a regular cell complex built from the molecule. Sheaf cohomology then supplies
chemically meaningful topological invariants: `dim H^0` counts non-bonding
orbitals, and the Hodge 1-Laplacian / `H^1` carries ring and holonomy
(Hückel–Möbius) content.

This repository contains everything needed to reproduce every number and figure
in the paper. All inputs are generated analytically by the code; there are no
external datasets.

## Contents

| Path | Description |
|------|-------------|
| `experiments/run_experiments.py` | E1 (Hamiltonian-to-sheaf embedding exact to machine precision), E2 (`dim H^0` = non-bonding-orbital counts over 11 conjugated molecules), E3 (*O*(3)-equivariance of the sheaf Laplacian), and the benzene worked example. Produces `fig_nonbonding.pdf`. |
| `experiments/e4_trainable.py` | E4: equivariant sheaf vs. coordinate-MLP learning curves on a directional HOMO–LUMO-gap target (rotation generalization and data efficiency). Produces `fig_learning_curve.pdf`. |
| `experiments/e5_h1_holonomy.py` | E5: `H^1` / Hodge 1-Laplacian checks — Betti number for the trivial sheaf, Z/2 holonomy for a Möbius ring, ring filling, and the Hodge cross-check `dim ker L_1 = dim H^1`. |
| `experiments/results.json` | Cached numerical results. |
| `main.tex`, `main.pdf` | Paper source and compiled PDF. |

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/run_experiments.py     # E1–E3 + benzene, writes fig_nonbonding.pdf
python experiments/e4_trainable.py         # E4, writes fig_learning_curve.pdf
python experiments/e5_h1_holonomy.py       # E5, prints all H^1 checks (asserts pass)
```

Tested with Python 3.13.

## Citation

```bibtex
@misc{harish2026ecsn,
  author = {Harish, Krishna},
  title  = {Equivariant Cellular Sheaves for Molecular Electronic Structure:
            Bridging Sheaf Cohomology and {E(3)}-Equivariant Hamiltonian Learning},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The paper text
and figures are © 2026 Krishna Harish.
