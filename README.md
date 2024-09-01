# LongRecipe

## Reproduction:

Before starting with the data preprocessing and model training, ensure that all necessary dependencies are installed. Use the following command to install the required packages:

`pip install -r requirements.txt`

### Data Preprocessing (Example: Llama3)

To begin, download the dataset represented by the Llama3 tokenizer from this link. After downloading, execute the following command to generate the position index files for different training approaches:


```
# Command to generate position index files (replace with actual command)
python generate_position_index.py --data_path <path_to_downloaded_data> --output_dir <output_directory>
```


### Model Training：

The model training process is divided into three distinct stages to effectively extend the context window of the LLM while maintaining its original capabilities.

#### Context Window Extension

In the first stage, we extend the context window using a dataset containing 1.7B tokens. The following command initiates this training stage:


```
accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 96 \
--learning-rate 5e-5 \
--epoch 1 \
--data_path $DATA_PATH_CONTEXT_EXTENSION \
--output-dir  ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL \
--seed 2027 \
--model $MODEL \
--seq-length $SEQ_LENGTH \
--target-length $TARGET_LENGTH \
--log-path $SETTING-$SEQ_LENGTH-$MODEL_NAME-$SUB_LABEL.log \
--setting $SETTING \
--rts-path $RTS_PATH \
--rss-path $RSS_PI_PATH \
--parallel_mode ulysses_attn \
--num_proc 5 \
--stage 1
```


Arguments Explanation:
* **--data_path**: Path to the dataset with Llama3-tokenized samples.
* **--model**: The base model used for training.
* **--seq-length**: The sequence length for training.
* **--target-length**: The target context window length.
* **--setting**: The training method, which could include FLT, RPES, PoSE, LongRecipe.
* **--rts-path**: Path to the PoSE right point set file.
* **--rss-path**: Path to the LongRecipe’s position index file.

Post-training, copy the tokenizer files to the output directory and remove any unnecessary files:

```
cp $MODEL/special_tokens_map.json ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL/stage_0
cp $MODEL/tokenizer_config.json ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL/stage_0
cp $MODEL/tokenizer.json ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL/stage_0
rm ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL/stage_0/model.safetensors
```

#### Stage 2: Training Annealing


In the second stage, we perform training annealing using both general and domain-specific data, gradually reducing the learning rate to zero. Approximately 100M tokens of data are used in this phase.
```
accelerate launch \
--config_file accelerate_configs/single_node_2.yaml \
train.py \
--data_path $DATA_PATH_ANNEALING \
--batch-size 1 \
--gradient-accumulate-every 96 \
--learning-rate 5e-6 \
--epoch 1 \
--output-dir  ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL \
--seed 2027 \
--model $STAGE_1_MODEL \
--seq-length $SEQ_LENGTH \
--target-length $TARGET_LENGTH \
--log-path $SETTING-$SEQ_LENGTH-$MODEL_NAME-$SUB_LABEL.log \
--setting $SETTING \
--rts-path $RTS_PATH \
--rss-path $RSS_PI_PATH \
--parallel_mode ulysses_attn \
--num_proc 10 \
--stage 2
```

Copy the updated tokenizer files to the output directory:

```
cp $MODEL/special_tokens_map.json ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL/stage_1
cp $MODEL/tokenizer_config.json ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL/stage_1
cp $MODEL/tokenizer.json ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL/stage_1
rm ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL/stage_1/model.safetensors
```

#### Stage 3: Model Merge

The final stage involves merging the original model with the fine-tuned model using an average weight strategy to enhance the model's foundational capabilities.

```
accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--output-dir ./output/$MODEL_NAME-$SETTING-80-$SEQ_LENGTH-$SUB_LABEL \
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
--stage 3
```

### Evaluation

Before conducting the evaluation of our method, you need to configure a new environment for it.

`pip install -r env_requirements.txt`

Once you train the model successfully, you can find the model files in the corresponding path you config

#### Evaluation for GSM8k, HumanEval, MMLU, LongBench and LooGLE
```
cd test_others
python infer.py \
— model_path XXXX \ ## path of model file
— tag XXX \ ## the name for model and method, such as llama_3_8b_pose_80_24
— ability all \  ## you can also select one of these options 'GSM8k', 'HumanEval', 'MMLU', 'LongBench', 'LooGLE'
— eval_time 3
```

#### Evaluation for Ruler
cd Ruler/scripts
```
sh run.sh model_name model_path synthetic
```
**model_name** is like llama3_8b_full_stage1_0820
**model_path** the path for model files
**synthetic** indicates the synthetic.yaml

#### Check the final evaluation scores for different benchmarks.


## Citation

If you find this repo helpful, please cite our paper as follows:

```
@article{hu2024longrecipe,
  title={LongRecipe: The Recipe for Efficient Long Context Generalization in
Large Languge Models},
  author={Zhiyuan Hu, Yuliang Liu},
  journal={arXiv preprint arXiv:2408.XXXXX},
  year={2023}
}
```
