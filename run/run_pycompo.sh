#!/bin/bash

CONFIG_FILE=/home/m/m300738/libs/pycompo/config/settings_ngc5004_pc02.yaml
RUNFILE1=/home/m/m300738/libs/pycompo/pycompo/api/get_features_full.py

# GET FEATURES
sbatch <<EOF
#!/bin/bash
#SBATCH --account=mh0731
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name="get_features"
#SBATCH --output=LOG/get_features.%j.out
#SBATCH --error=LOG/get_features.%j.out
#SBATCH --export=ALL

source ~/.bashrc
conda activate TRR181L4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "${RUNFILE1}" "${CONFIG_FILE}"
EOF
