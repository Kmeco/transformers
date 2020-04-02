<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/transformers_logo_name.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
</p>

<h3 align="center">
<p>State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch
</h3>

This repo has been forked from huggingface/transfomrers. Please refer to the links above for more information and general instructions.

### Project objective
Fine-tune GPT-2 to build an abstractive summarisation model.

### Dataset
We are using the [CNN/DailyMail](https://github.com/abisee/cnn-dailymail) dataset.

Our model was trained on 200k samples and evaluated on 11k test examples that can be found [here](https://drive.google.com/drive/folders/1wNbUzt1aWi1XUCWZCmImfA_9dtQ8aQjj?usp=sharing).

### Train script

```
wandb login 6d7c61fffdcce709c75c149002e70c51001718ed

cd /content/transformers/examples

python3 run_language_modeling.py \
    --model_type=gpt2 \
    --model_name_or_path='/content/drive/My Drive/models/ex10-cnn-output-long/checkpoint-latest' \
    --output_dir='/content/drive/My Drive/models/ex10-cnn-output-long' \
    --overwrite_output_dir \
    --num_train_epochs=3 \
    --per_gpu_train_batch_size=7 \
    --block_size=512 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --seed=1234 \
    --one_by_one \
    --save_total_limit=1 \
    --save_steps=1000 \
    --logging_steps=5000 \
    --evaluate_during_training \
    --wandb "ucl-nlp-project"
```

### Evaluation

```
cd transformers/examples

python run_batch_generation.py \
    --model_type=gpt2 \
    --output_dir='/content/output_batch_cnn_R' \
    --model_name_or_path='/content/drive/My Drive/models/ex10-cnn-output-long/final' \
    --eval_data_file='/content/pass_forward_R' \
    --length=120 \
    --block_size=512 \
    --stop_token='<EOD>' \
    --per_gpu_eval_batch_size=1
```
