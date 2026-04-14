#!/bin/bash

analysis_identifier=${1}
node_size=${2:-"256G"}
if [ -z "$analysis_identifier" ]; then
    echo "Usage: $0 <analysis_identifier>"
    exit 1
fi

export ACCOUNT=bm1500
export CONFIG_FILE=/home/m/m300738/libs/pycompo/config/settings_${analysis_identifier}.yaml
export RUNFILE1=/home/m/m300738/libs/pycompo/pycompo/drivers/combine_feature_props.py
export RUNFILE2=/home/m/m300738/libs/pycompo/pycompo/drivers/get_composites.py


# COMBINE FEATURE PROPS
jobid1=$(sbatch --parsable --account=${ACCOUNT} <<EOF
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name="combine_feature_props"
#SBATCH --output=LOG/combine_feature_props.%j.out
#SBATCH --error=LOG/combine_feature_props.%j.out
#SBATCH --export=ALL

source ~/.bashrc
micromamba activate TRR181L4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "${RUNFILE1}" "${CONFIG_FILE}"
EOF
)

# GET COMPOSITES
jobid2=$(sbatch --parsable --constraint=${node_size} --dependency=afterok:${jobid1} --account=${ACCOUNT} <<EOF
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name="get_composites"
#SBATCH --output=LOG/get_composites.%j.out
#SBATCH --error=LOG/get_composites.%j.out
#SBATCH --export=ALL

source ~/.bashrc
micromamba activate TRR181L4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "${RUNFILE2}" "${CONFIG_FILE}"
EOF
)
