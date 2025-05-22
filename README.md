
<div align="center">
   <h1>ExSearch</h1>
</div>
</div>

If you like our project, please give us a star ⭐ on GitHub for the latest update.


# Environment

1. Install the necessary Python libraries by running the following commands.

```shell
conda create -n exsearch python=3.10
conda activate exsearch
pip install -r requirements.txt 
```

> You can customize your own `torch`, `vllm` and `transformers` version based on the backbone LLM you want to use.
> For Qwen and Mistral, we suggest `vllm=0.6.3`, `torch=2.4.0+cu118` and `transformers=4.45.0`.

2. Set up the retrieval module. We follow previous work and use the Wikipedia as our document corpus, which can be found in [DPR](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py) repo ([Link](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz)).
We use the [ColBERT](https://github.com/stanford-futuredata/ColBERT/tree/main) as the retrieval model to pair each query with top-20 documents. The pre-trained ColBERT checkpoint can be downloaded in either its official repo or its [link](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz).
You can deploy the ColBERT retrieval or other customized retrieval model in your local environment to pre-process the dataset. 

In this project, the code `retrieval` folder is directly copied from [ColBERT](https://github.com/stanford-futuredata/ColBERT/tree/main). You can following the `README.md` in `retrieval` to set up the retriever.


# Train your Search Language Models via Expectation-Maximization

## Warmup training
Before the iterative E&M training, we first train the LLM with warm-up dataset, similar to the cold start process in previous work.
In this warmup training, the LLM is trained on a small set of synthetic data, learning basic search and answer generation pattern.

```shell
PROCEDURE=sft CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup torchrun   --nproc_per_node=8 --master_port=11021 ./src/run.py \
--model_name_or_path HUGGINGFACE_MODEL_NAME_OR_LOCAL_MODEL_CHECKPOINT \
--dataset_name_or_path  WARM_UP_DATA_PATH \
--deepspeed ./src/script/ds_z3_config.json \
--output_dir ./mistral24_100 \
--overwrite_cache True \
--warmup_ratio 0.1 \
--report_to wandb \
--run_name test_run \
--logging_steps 1 \
--cutoff_len 8192 \
--max_samples 200000 \
--save_steps  1000 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--learning_rate 2.0e-6 \
--num_train_epochs 2 \
--lr_scheduler_type cosine \
--bf16 True & \
--resume_from_checkpoint /root/paddlejob/workspace/env_run/output/SearchAgent/agent2_musique/checkpoint-200  &
```

Set the `WARM_UP_DATA_PATH` to your own data path such as `./data/hotpot-traj.100.json`.


## E-step and M-step Training 

### E-step: Trajectory Exploration

```shell
mkdir ./log
PROCEDURE=inference CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python ./src/run.py \
--model_name_or_path HUGGINGFACE_MODEL_NAME_OR_LOCAL_MODEL_CHECKPOINT \
--input_file TRAINING_OR_EVALUATION_FILE_PATH \
--output_dir ./log \
--left 0 \
--right 10000
```
Set the `TRAINING_OR_EVALUATION_FILE_PATH` to your own training data file or evaluation data file, such as `data/eval_data/hotpotqa_dev.json`.

Once your finish the above command, the output of LLM is stored into a local file. Please use this file as the argument for `OUTPUT_FILE` below.
```shell
PROCEDURE=entropy CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./src/run.py \
--model_name_or_path HUGGINGFACE_MODEL_NAME_OR_LOCAL_MODEL_CHECKPOINT \
--inference_file OUTPUT_FILE  \
--output_dir ./log  \
--epo EPO
```
Here the `EPO` denotes the current training iteration, such as `1` for the $1^{st}$ iteration and `2` for the $2^{nd}$ iteration.

### M-step: Re-weighted Trajectory Learning

```shell
PROCEDURE=align CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup torchrun   --nproc_per_node=8  --master_port=11021 ./src/run.py \
--model_name_or_path HUGGINGFACE_MODEL_NAME_OR_LOCAL_MODEL_CHECKPOINT \
--dataset_name_or_path  TRAINING_DATA \
--deepspeed ./src/script/ds_z3_config.json \
--output_dir OUTPUT_CHECKPOINT_FOLDER \
--overwrite_cache True \
--warmup_ratio 0.1 \
--report_to wandb \
--run_name test_run \
--logging_steps 1 \
--cutoff_len 8192 \
--max_samples 300000 \
--save_steps  200 \
--per_device_train_batch_size  2 \
--gradient_accumulation_steps 16 \
--learning_rate 2.0e-6 \
--num_train_epochs 2 \
--lr_scheduler_type cosine \
--bf16 True 
```
Note that:
1. Adding the `--resume_from_checkpoint OUTPUT_CHECKPOINT_FOLDER` argument if the training is broken and you want to continue the training.
2. You can customize the arguments like `per_device_train_batch_size`, `gradient_accumulation_steps` and `training epoch` based on your own computational resource.

# Acknowledgement
We sincerely thank prior work, including [ColBERT](https://github.com/stanford-futuredata/ColBERT/tree/main) and [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main).


# Citation
```txt

```