# First Aid Llama 🦙🚑

A fine-tuned version of Llama 3.1 (8B) specialized in medical reasoning and first aid advice.

## 📖 Overview
This model was fine-tuned using [Unsloth](https://github.com/unslothai/unsloth) on the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset. It is designed to answer medical queries with detailed Chain-of-Thought (CoT) reasoning.

**Base Model:** `unsloth/Meta-Llama-3.1-8B`
**Dataset:** Medical O1 Reasoning SFT
**Method:** LoRA (Low-Rank Adaptation) with 4-bit quantization

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/First-Aid-Llama.git](https://github.com/YOUR_USERNAME/First-Aid-Llama.git)
   cd First-Aid-Llama