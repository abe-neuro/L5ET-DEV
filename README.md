# Developmental emergence of layer 5 extratelencephalic neuron areal diversity

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abe-neuro/L5ET-DEV.git/HEAD?urlpath=rstudio)

## Contents

-   [Overview](#overview)
-   [Repo Contents](#repo-contents)
-   [Requirements](#requirements)
-   [Links](#Links)
-   [Citation](#citation)

# Overview

The repository L5ET-DEV is a collection of `R` scripts and data to reproduce the main data panels of Abe et al.,2024.

# Repo Contents

-   [scripts](./scripts): `R` scripts for plotting the main data figures.
-   [data](./data): data and helper tables.

# Requirements

## Data

Data can be downloaded from GEO server:

-   snRNA-seq of developing L5 ET neurons

-   MAPseq mapping of L5 ET projections

Download and save the R object (.RDS) locally. Change the path in the script to this location to load the object.

## Hardware Requirements

We recommend a computer with the following specs:

-   RAM: more than 16 GB

-   CPU: 4+ cores, 3.3+ GHz/core

OS Requirements: The scripts were tested on Windows and Mac OS.

Further, CRAN package should be compatible with Windows, Mac, and Linux operating systems.

The user should install `R` version 4.4.1 or higher, and several packages set up from CRAN or from source (see scripts for detail).

# Links

ShinyServer for transcriptional landscapes of Abe et al., 2024: <http://genebrowser.unige.ch/L5ET_shiny-v3/>

<https://neurocenter-unige.ch/research-groups/denis-jabaudon/>

<https://github.com/pradosj/docker_sindbis>

<https://github.com/awaisj14/CellOracle_ET/tree/main/Pseudobulk_ATAC_files_Yao_2021>

# Citation

###Research article

*Published soon*

###Preprint

**Developmental molecular controls over arealization of descending cortical motor pathways**

Philipp Abe, Adrien Lavalley, Ilaria Morassut, Esther Klingler, Antonio J. Santinha, Randall J. Platt, Denis Jabaudon

*bioRxiv* 2023.06.29.546438; doi: <https://doi.org/10.1101/2023.06.29.546438>
