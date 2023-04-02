#!/bin/bash

# compute node don't have write permisson and ability to connect internet
export WANDB_DIR=/gpfs/alpine/scratch/lfsm/csc499/wandb
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/
export WANDB_MODE=dryrun

# set up module and other env variable
# this env and shell are built following 
# https://docs.google.com/document/d/1gyMVFonqgTRToaHckEPYjGTX-06cZcO04VoT-GHoXAc/edit#heading=h.zh8lrk889uso


