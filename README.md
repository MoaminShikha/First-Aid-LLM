# First Aid Llama 🦙🚑

A fine-tuned version of Llama 3.1 (8B) specialized in medical reasoning and first aid advice.

> ⚠️ This project is for research and prototyping only. It is **not** a medical device. Always consult licensed medical professionals and follow local emergency protocols.

## 📖 Overview
This model was fine-tuned using [Unsloth](https://github.com/unslothai/unsloth) on the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset. It is designed to answer medical queries with detailed Chain-of-Thought (CoT) reasoning.

- **Base Model:** `unsloth/Meta-Llama-3.1-8B`
- **Dataset:** `FreedomIntelligence/medical-o1-reasoning-SFT`
- **Method:** LoRA (Low-Rank Adaptation) with 4-bit quantization

## 🛠️ Quickstart

### 1) Setup

```bash
git clone https://github.com/YOUR_USERNAME/First-Aid-Llama.git
cd First-Aid-Llama
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Hardware: An NVIDIA GPU with ≥24 GB VRAM is recommended for training; inference on smaller GPUs/CPUs is possible with quantized weights but may be slower.

### 2) Configure

Adjust values in `configs/default.yaml` to change the base model, LoRA hyperparameters, dataset limits, or generation settings. Key knobs:
- `model.name` – base checkpoint (4-bit quantized by default)
- `splits` – validation/test split sizes
- `training.max_steps`, `training.learning_rate` – training schedule
- `inference` – generation and safety prefix

### 3) Train

Train with the provided script (uses Unsloth + TRL under the hood):

```bash
python scripts/train.py --config configs/default.yaml --output_dir outputs
```

The script:
- Cleans and splits the dataset with length filters and duplicate removal
- Formats prompts with the Llama 3.1 chat template
- Applies LoRA adapters
- Logs split sizes and saves the best checkpoint + tokenizer to `outputs/`

### 4) Inference

Run inference with guardrails and attention masks to avoid generation warnings:

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --model_path outputs \
  --prompt "What should I do if someone is choking?"
```

You can also pass a JSON list of chat messages via `--messages_json` for multi-turn inputs.

### 5) Exporting

The notebook (and scripts) demonstrate saving LoRA adapters locally. To export GGUF or push to the Hugging Face Hub, use the Unsloth `save_pretrained_gguf` / `push_to_hub_gguf` utilities shown in the notebook—remember to replace placeholder repo IDs and tokens.

## 🧰 Repository layout

- `configs/default.yaml` – reproducible run configuration (model, LoRA, dataset, training, inference)
- `first_aid_llm/` – reusable Python package
  - `config.py` – typed config loader
  - `data.py` – dataset loading, cleaning, splitting, and prompt formatting
  - `training.py` – model/tokenizer setup and trainer builder
  - `inference.py` – safe generation helper with attention masks
- `scripts/train.py` – CLI entrypoint for training
- `scripts/infer.py` – CLI entrypoint for inference
- `First_Aid_LLM.ipynb` – end-to-end walkthrough (kept for interactive use)

## ✅ Safety & quality checklist

- Adds a safety disclaimer prefix to inference responses
- Uses attention masks during generation to avoid warnings and unpredictable behavior
- Filters short/long/duplicate dataset rows before training
- Tracks train/val/test splits for reproducibility

## 🔬 Testing the setup

Lightweight sanity check (CPU-only) that verifies imports and config parsing:

```bash
python - <<'PY'
from first_aid_llm.config import AppConfig
cfg = AppConfig.from_yaml('configs/default.yaml')
print(cfg.run_name)
PY
```

Full training requires a supported NVIDIA GPU and will take time/resources; use `training.max_steps` to limit iterations for quick experiments.
