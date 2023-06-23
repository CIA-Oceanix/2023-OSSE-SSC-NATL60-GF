# 2023 OSSE SSC NATL60 GF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8076318.svg)](https://doi.org/10.5281/zenodo.8076318)

This repository is linked to the following research paper:

> Fablet, R, Chapron, B, Sommer, J, Sévellec, F. Inversion of sea surface
> currents from satellite-derived SST-SSH synergies with 4DVarNets.
> [⟨arxiv⟩](https://doi.org/10.48550/arXiv.2211.13059)

See the [notebook](notebook.ipynb) for results.


## Data

The studied oceanographic region is located between (33°N, 65°W) and (43°N, 55°W). The considered simulation dataset relies on a nature run of the NATL60 configuration [Ajayi et al., 2020] of the NEMO
(Nucleus for European Modeling of the Ocean) model [Madec et al., 2022]. The simulation is run without tidal forcing.

In the notebook, only the evaluation period is used to compute the scores (from 2012-10-22 to 2012-12-03).

The data are distributed by AVISO+ (see [here](https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60/wiki/AVISO---account-creation) to download) in netCDF4 format. The data are also available (temporarily) [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/ocean-data-challenges/dc_data1/catalog.html) or by executing the following commands:

Observation data (~400 Mb):
```bash
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_obs.tar.gz
```

Reference data (~11 Gb):
```bash
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_ref.tar.gz
```

See [SSH Mapping Data Challenge 2020a](https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60) README for more information.


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
