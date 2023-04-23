#!/usr/bin/env bash

python scripts/extract.py esm1b_t33_650M_UR50S data/cath-data/all_data.fasta esm_embeddings/ --include per_tok
