SAVE_PATH='./output'
OUTPUT_PATH="./output/audio_feature"
AUDIO_PATH='/app/lrs3'

HYDRA_FULL_ERROR=1 python ./src/core/preprocess/audio_feature_extraction/main.py \
save_path="$SAVE_PATH" \
Trainer.devices="1" \
module.save_path="$OUTPUT_PATH" \
module.backbone.device="cuda:1" \
module.dataset.preprocess.video_path="$AUDIO_PATH"