#!/bin/bash

CONFIG_FILE=/home/m/m300738/libs/pycompo/config/settings_ngc5004_pc04.yaml
RUNFILE1=/home/m/m300738/libs/pycompo/pycompo/api/get_features.py
RUNFILE2=/home/m/m300738/libs/pycompo/pycompo/api/combine_feature_props.py
RUNFILE3=/home/m/m300738/libs/pycompo/pycompo/api/get_composites.py

# GET FEATURES
jobid1 = $(sbatch --parsable <<EOF
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
)


# COMBINE FEATURE PROPS
jobid2 = $(sbatch --parsable --dependency=afterok:${jobid1} <<EOF
#!/bin/bash
#SBATCH --account=mh0731
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name="combine_feature_props"
#SBATCH --output=LOG/combine_feature_props.%j.out
#SBATCH --error=LOG/combine_feature_props.%j.out
#SBATCH --export=ALL

source ~/.bashrc
conda activate TRR181L4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "${RUNFILE2}" "${CONFIG_FILE}"
EOF
)

# GET COMPOSITES
jobid3 = $(sbatch --parsable --dependency=afterok:${jobid2} sbatch <<EOF
#!/bin/bash
#SBATCH --account=mh0731
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name="get_composites"
#SBATCH --output=LOG/get_composites.%j.out
#SBATCH --error=LOG/get_composites.%j.out
#SBATCH --export=ALL

source ~/.bashrc
conda activate TRR181L4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "${RUNFILE3}" "${CONFIG_FILE}"
EOF
)