# Anti Distillation

## Recommended Environment

For the long-term "SFT training + vLLM 0.8.5 decoding" workflow, the repo now includes a unified environment manifest:

- `envs/adistill-unified-vllm085.yml`

This manifest intentionally uses the stricter `vLLM 0.8.5` binary stack as the anchor
(`torch 2.6.0` / `transformers 4.52.4`) and then layers the current HF SFT
dependencies on top. That is a safer direction than trying to install `vllm==0.8.5`
into a newer training-only stack such as `torch 2.10.x`.

`envs/adistill-unified-vllm085.yml` is a standard Conda environment manifest:

- `name`: the environment name created by Conda
- `channels`: which Conda package channels to resolve from
- `dependencies`: Conda-managed packages
- `pip`: pip-managed packages installed after the Conda part is ready

Create the environment:

```bash
conda env create -f envs/adistill-unified-vllm085.yml
conda activate adistill-unified
```

Update an existing environment after editing the YAML:

```bash
conda env update -n adistill-unified -f envs/adistill-unified-vllm085.yml --prune
```

Here:

- `-f` means "read the environment definition from this file"
- `-n` means "target this Conda environment name"
- `--prune` means "remove packages that are no longer listed in the YAML"

## Current SFT Layout

The training code is now organized under `src/sft/`:

- `src/sft/config.py`: training configuration and CLI arguments
- `src/sft/data.py`: GSM8K formatting, dataset, and collator
- `src/sft/distributed.py`: distributed/FSDP runtime setup
- `src/sft/trainer.py`: model loading, train loop, and evaluation
- `src/train_sft.py`: main training entry point

`src/phaseB_infer.py` is kept as a thin compatibility wrapper and now forwards to `src/train_sft.py`.

The default dataset backend is now Hugging Face with `openai/gsm8k`.
ModelScope is still supported as an optional backend.

## vLLM Offline Demo

To keep inference experiments decoupled from the SFT training code, the native
vLLM offline demo now lives under `examples/vllm_offline/`:

- `examples/vllm_offline/run_qwen_offline.py`: minimal offline inference script
- `examples/vllm_offline/sample_prompts.txt`: line-based prompt batch example
- `examples/vllm_offline/README.md`: install notes, commands, and workflow explanation

## How To Run

Single process:

```bash
python src/train_sft.py
```

Multi-GPU with FSDP:

```bash
torchrun --nproc_per_node=4 src/train_sft.py
```

Common 2-GPU FSDP run:

```bash
CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 src/train_sft.py
```

Common overrides:

```bash
python src/train_sft.py --max-length 384 --train-batch-size 1 --num-epochs 2
```

Quick FSDP debug run:

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=4 src/train_sft.py --log-every 1 --eval-every 0 --max-steps 1 --debug-fsdp
```

If `modelscope` is not installed yet, install it in your training environment first:

```bash
pip install modelscope
```

To temporarily switch back to Hugging Face datasets:

```bash
python src/train_sft.py --dataset-backend huggingface --dataset-name openai/gsm8k --dataset-namespace ""
```

To disable ModelScope remote-code execution explicitly:

```bash
python src/train_sft.py --disable-modelscope-trust-remote-code
```
