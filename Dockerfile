## Declares build arguments
ARG NB_USER=rstudio
ARG NB_UID=1001

FROM rocker/binder:4.4.1

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
RUN install2.r --error --skipinstalled -n 4 torch luz BiocManager
RUN Rscript -e 'torch::install_torch()'


## Copy files to home folder
COPY --chown=${NB_USER} ./ /home/${NB_USER}/
