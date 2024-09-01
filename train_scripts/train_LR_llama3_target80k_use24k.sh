
SEQ_LENGTH=24000
TARGET_LENGTH=80000
SETTING='LongRecipe'
MODEL_NAME=llama3_8b
RTS_PATH=./output/RTS.pkl
RSS_PI_PATH=./output/LR_PI.pkl
SUB_LABEL='LR_target80k_use24k'
DATA_PATH_1='jsonl_file'
DATA_PATH_2='replay_dataset'
MODEL='model_path'
# --parallel_mode: data_parallel; 

accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 96 \
--learning-rate 5e-5 \
--epoch 1 \
--data_path $DATA_PATH_1 \
--output-dir  ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL \
--seed 2027 \
--model $MODEL \
--seq-length $SEQ_LENGTH \
--target-length $TARGET_LENGTH \
--log-path $SETTING-$SEQ_LENGTH-$MODEL_NAME-$SUB_LABEL.log \
--setting $SETTING \
--rts-path $RTS_PATH \
--rss-path $RSS_PI_PATH \
--parallel_mode data_parallel \
--num_proc 5 \
--stage 0

cp $MODEL/special_tokens_map.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_0
cp $MODEL/tokenizer_config.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_0
cp $MODEL/tokenizer.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_0
rm ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_0/model.safetensors


accelerate launch \
--config_file accelerate_configs/single_node_2.yaml \
train.py \
--data_path $DATA_PATH_2 \
--batch-size 1 \
--gradient-accumulate-every 96 \
--learning-rate 5e-6 \
--epoch 1 \
--output-dir  ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL \
--seed 2027 \
--model $MODEL \
--seq-length $SEQ_LENGTH \
--target-length $TARGET_LENGTH \
--log-path $SETTING-$SEQ_LENGTH-$MODEL_NAME-$SUB_LABEL.log \
--setting $SETTING \
--rts-path $RTS_PATH \
--rss-path $RSS_PI_PATH \
--parallel_mode data_parallel \
--num_proc 5 \
--stage 1

cp $MODEL/special_tokens_map.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_1
cp $MODEL/tokenizer_config.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_1
cp $MODEL/tokenizer.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_1
rm ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_1/model.safetensors


accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--output-dir  ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL \
--seed 2027 \
--model $MODEL \
--log-path $SETTING-$SEQ_LENGTH-$MODEL_NAME-$SUB_LABEL.log \
--stage 2

cp $MODEL/special_tokens_map.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_2
cp $MODEL/tokenizer_config.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_2
cp $MODEL/tokenizer.json ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_2
# rm ./output/$MODEL_NAME-$SETTING-$SEQ_LENGTH-$SUB_LABEL/stage_2/model.safetensors

