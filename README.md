# Anti Distillation

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
