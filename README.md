# Molecular programs guiding arealization of distinct descending cortical pathways

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

-   snRNA-seq of developing L5 ET neurons: [GSE270951](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270951)

-   MAPseq mapping of L5 ET projections: [GSE271067](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi)

We recommend downloading the R object (.RDS file). This file contains the quality controlled and processed data. To load the object, change the path in the R script to the download location.

## Hardware Requirements

We recommend a computer with the following specs:

-   RAM: more than 16 GB

-   CPU: 4+ cores, 3.3+ GHz/core

OS Requirements: The scripts were tested on Windows and Mac OS.

Further, CRAN package should be compatible with Windows, Mac, and Linux operating systems.

The user should install `R` version 4.4.1 or higher, and several packages set up from CRAN or from source (see scripts for detail).

# Links

ShinyServer for transcriptional landscapes of Abe et al., 2024: <https://genebrowser.unige.ch/L5ETdev>

<https://neurocenter-unige.ch/research-groups/denis-jabaudon/>

<https://github.com/pradosj/docker_sindbis>

<https://github.com/awaisj14/CellOracle_ET/tree/main/Pseudobulk_ATAC_files_Yao_2021>

# Citation

### Research article

**Molecular programs guiding arealization of descending cortical pathways.**

Philipp Abe, Adrien Lavalley, Ilaria Morassut, Antonio J. Santinha, Sergi Roig-Puiggros, Awais Javed, Esther Klingler, Natalia Baumann, Julien Prados, Randall J. Platt & Denis Jabaudon.

Nature (2024). https://doi.org/10.1038/s41586-024-07895-y


### Preprint

**Developmental molecular controls over arealization of descending cortical motor pathways**

Philipp Abe, Adrien Lavalley, Ilaria Morassut, Esther Klingler, Antonio J. Santinha, Randall J. Platt, Denis Jabaudon

*bioRxiv* 2023.06.29.546438; doi: <https://doi.org/10.1101/2023.06.29.546438>
