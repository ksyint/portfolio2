SAVE_PATH='./output/unsupervised/PCL'

HYDRA_FULL_ERROR=1 python src/core/train/unsupervised/codebook/main.py \
save_path="$SAVE_PATH" \
module.batch_size=2