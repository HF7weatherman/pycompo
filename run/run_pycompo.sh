#!/bin/bash

analysis_identifier=${1}
node_size=${2:-"256G"}
if [ -z "$analysis_identifier" ]; then
    echo "Usage: $0 <analysis_identifier>"
    exit 1
fi

export CONFIG_FILE=/home/m/m300738/libs/pycompo/config/settings_${analysis_identifier}.yaml
export RUNFILE1=/home/m/m300738/libs/pycompo/pycompo/api/get_features.py
export RUNFILE2=/home/m/m300738/libs/pycompo/pycompo/api/combine_feature_props.py
export RUNFILE3=/home/m/m300738/libs/pycompo/pycompo/api/get_composites.py

# GET FEATURES
jobid1=$(sbatch --parsable --constraint=${node_size} <<EOF
#!/bin/bash
#SBATCH --account=mh0731
#SBATCH --partition=compute
#SBATCH --mem=0
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
jobid2=$(sbatch --parsable --constraint=${node_size} --dependency=afterok:${jobid1} <<EOF
#!/bin/bash
#SBATCH --account=mh0731
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
conda activate TRR181L4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "${RUNFILE2}" "${CONFIG_FILE}"
EOF
)

# GET COMPOSITES
jobid3=$(sbatch --parsable --constraint=${node_size} --dependency=afterok:${jobid2} <<EOF
#!/bin/bash
#SBATCH --account=mh0731
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
conda activate TRR181L4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "${RUNFILE3}" "${CONFIG_FILE}"
EOF
)

squeue -u $USER