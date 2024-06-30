FROM rocker/binder:4.4.1

## Declares build arguments
ARG NB_USER=rstudio
ARG NB_UID=1001

ENV DEBIAN_FRONTEND=noninteractive
USER root

## Install linux packages
RUN apt-get update --fix-missing > /dev/null \
        && apt-get install --yes \
          libncurses5-dev \
        && apt-get clean > /dev/null \
        && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

## Install R packages
RUN install2.r --error --skipinstalled -n 4 devtools torch luz BiocManager Seurat scales wesanderson scico RColorBrewer reshape2
RUN Rscript -e 'BiocManager::install(c("DelayedArray","SingleCellExperiment","HDF5Array","scuttle"))'
RUN Rscript -e 'torch::install_torch();devtools::install_github("BioinfoSupport/scml",upgrade = FALSE)'


## Copy files to home folder
COPY --chown=${NB_USER} ./ /home/${NB_USER}/
