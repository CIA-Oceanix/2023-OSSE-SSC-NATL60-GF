# 2023 OSSE SSC NATL60 GF

This repository is linked to the following research paper:

> Fablet, R, Chapron, B, Sommer, J, Sévellec, F. Inversion of sea surface
> currents from satellite-derived SST-SSH synergies with 4DVarNets.
> [⟨arxiv⟩](https://doi.org/10.48550/arXiv.2211.13059)


## Leaderboard

| Method | Data | $\lambda_{x,u}$ | $\lambda_{x,v}$ | $\lambda_{t,u}$ | $\lambda_{t,v}$ | $\tau_{u,v}$ | $\tau_{\text{vort}}$ | $\tau_{\text{div}}$ | $\tau_{\text{strain}}$ |
| - | - | - | - | - | - | - | - | - | - |
| Ground truth | SSH | 0.36 | 0.17 | 19.6 | 11.2 | 97 | 96.3 | -1.0 | 92.1 |
| DUACS | SSH | 1.72 | 1.24 | 12.6 | 11.6 | 83.7 | 53.5 | -0.5 | 24.8 |
| U-Net | SSH | 1.39 | 1.22 | 9.1 | 10.3 | 89.1 | 72.3 | -3.0 | 65.0 |
| U-Net | SSH-SST | 1.33 | 0.90 | 4.0 | 4.2 | 92.6 | 79.4 | 19.5 | 72.0 |
| 4DVarNet | SSH | 0.9 | 0.7 | 4.3 | 5.6 | 94.0 | 86.1 | 12.1 | 81.3 |
| 4DVarNet | SSH-SST | **0.76** | **0.61** | **2.7** | **2.5** | **97.4** | **92.1** | **46.9** | **87.2** |

Where:

- $\lambda_{x,u}$ (degrees): minimum spatial scale resolved for the zonal velocity
- $\lambda_{x,v}$ (degrees): minimum spatial scale resolved for the meridional velocity
- $\lambda_{t,u}$ (days): minimum time scale resolved for the zonal velocity
- $\lambda_{t,v}$ (days): minimum time scale resolved for the meridional velocity
- $\tau_{u,v}$ (%): explained variance of the reconstructed SSC
- $\tau_{\text{vort}}$ (%): explained variance of the vorticity
- $\tau_{\text{div}}$ (%): explained variance of the divergence
- $\tau_{\text{strain}}$ (%): explained variance for the strain

The best results are in bold.


## Installation & Execute

If you want to modify/run the notebook, please install first the
environment:

```sh
conda create -n ssc
conda activate ssc
conda env update -f environment.yaml
```

Then you can activate the environment and open the notebook with your
favourite editor.

The environment already includes Jupyter Lab and Jupyter Notebook.

```sh
conda activate ssc
jupyter lab  # or `jupyter notebook`
```
