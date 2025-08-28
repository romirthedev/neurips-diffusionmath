<div  align="center">
    <h1>d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning</h1>
  <p>A two-stage approach combining masked SFT with <i>diffu</i>-GRPO—a novel policy gradient method based on GRPO that features efficient log probability estimation designed for masked dLLMs—to scale reasoning capabilities in pre-trained diffusion Large Language Models</p>
</div>



![Results](media/pull_fig.png)

![Results](media/sota.png)

<div align="center">
  <hr width="100%">
</div>

**🔄Updates:**

* 05-04-2025: We released the diffu-GRPO and eval code.
* 04-11-2025: We released [our paper](https://dllm-reasoning.github.io/media/preprint.pdf) and [project page](https://dllm-reasoning.github.io). Additionally, the SFT code was open-sourced.

<div align="center">
  <hr width="100%">
</div>



## Environment Setup

To setup the environment, run;
```
conda env create -f env.yml
conda activate d1
```


## SFT

We open-source our code to perform completion-only masked SFT for dLLMs. We implement the algorithm proposed in [LLaDA](https://github.com/ML-GSAI/LLaDA), and also provide it below for completeness.

![SFT Algorithm](media/algorithm_sft.png)

The framework follows a similar interface to 🤗 Transformers. `dLLMTrainer` subclasses `Trainer` and overrides the loss computation to implement the diffusion loss. `dLLMDataCollator` extends `DefaultDataCollator` by incorporating a forward noising process applied to each training batch. Additionally, we provide a custom torch dataset, `dLLMSFTDataset`, tailored for completion-only SFT of dLLMs.

To preprocess and tokenize your dataset, you will need to modify `preprocess_dataset`. Presently, it works with the s1K dataset.

SFT results can be reproduced with the command,
```bash
# First go to the SFT directory
cd SFT

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ddp_config.yaml --main_process_port 29500 --num_processes 2 sft_train.py --grad_accum_steps 4 --batch_size 1 --num_epochs 20 
# this results in effective batch size of 8 = 1 * 2 * 4, where 2 is the number of gpus.
```


## _diffu_-GRPO

The code is inside the `diffu-grpo` directory.

- `diffu-grpo/slurm_scripts` contains the slurm scripts we used to run the RL experiments
- Example bash script for running the RL experiment:
  ```bash
  cd diffu-GRPO
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
  ```

RL training curves across four reasoning tasks, with models initialized from Llada-Instruct (with and without SFT on s1K):

![RL Curves](media/rl_curves_train.png)



## Evaluation

The evaluation code is inside the `eval` directory.

- Run with `bash run_eval.sh`
- The evaluation file will only save the generations; use the parser to calculate accuracy
- For example, baseline generations are in the `eval_baselines` directory. Use `python parse_and_get_acc.py` to print the accuracy.


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{zhao2025d1,
  title={d1: Scaling reasoning in diffusion large language models via reinforcement learning},
  author={Zhao, Siyan and Gupta, Devaansh and Zheng, Qinqing and Grover, Aditya},
  journal={arXiv preprint arXiv:2504.12216},
  year={2025}
}
```
## How to Run - Addition

Here are the full steps to run your training on TACC after removing the dataset preprocessing split modification step (step 7):

***

### 1. Connect to TACC via SSH

```bash
ssh romirpatel@vista.tacc.utexas.edu
```

***

### 2. Create project folder inside scratch on TACC

```bash
mkdir -p /scratch/10936/romirpatel/d1_training
```

***

### 3. Copy your local repo to TACC Scratch using SCP

From your **local machine terminal**, run:

```bash
scp -r /Users/romirpatel/d1 romirpatel@vista.tacc.utexas.edu:/scratch/10936/romirpatel/d1_training/
```

This copies your entire `d1` folder to the scratch folder on TACC.

***

### 4. SSH back to TACC and navigate to your repo folder

```bash
ssh romirpatel@vista.tacc.utexas.edu
cd /scratch/10936/romirpatel/d1_training/d1/SFT
```

***

### 5. Setup Conda environment on TACC

If you have not done so yet, create and activate your conda environment:

```bash
conda env create -f ../env.yml
conda activate d1
```

***

### 6. Set Hugging Face cache environment variables

```bash
export HF_HOME=$SCRATCH/huggingface_cache
export HF_HUB_CACHE=$HF_HOME
mkdir -p $HF_HOME
```

Caches will be stored on scratch to avoid permission issues and allow fast access.

***

### 7. Run your training

Run the training script with your usual parameters, for example:

```bash
python sft_train.py \
  --model_name GSAI-ML/LLaDA-8B-Instruct \
  --batch_size 1 \
  --max_length 2048 \
  --num_epochs 1 \
  --learning_rate 1e-5 \
  --output_dir $SCRATCH/d1_training/output \
  --job_name debug-run \
  --train_data AI-MO/NuminaMath-LEAN \
  --debugging
```

***

### 8. Copy output to persistent storage after training finishes

```bash
cp -r $SCRATCH/d1_training/output /home1/10936/romirpatel/d1_outputs/
```

***

### 9. Exit the session

```bash
exit
```

***




