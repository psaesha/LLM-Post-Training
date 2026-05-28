# LLM Post-Training — Hindi Instruction Tuning

Post-training `Qwen/Qwen2.5-1.5B` for Hindi instruction-following using QLoRA → SFT → DPO.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

---

## Goal

The base Qwen2.5-1.5B model produces poor Hindi output — repetitive, off-topic, or transliterated responses. This project fine-tunes it on a curated Hindi instruction dataset using parameter-efficient QLoRA, then applies DPO to align outputs with human preferences.

**Target model on HuggingFace Hub:** [`psaesha/llm-post-trained-qwen2.5-1.5B`](https://huggingface.co/psaesha/llm-post-trained-qwen2.5-1.5B)

---

## Repo Structure

```
.
├── configs/
│   └── sft.yaml          # SFT hyperparameters and LoRA config
├── notebooks/
│   └── colab_runner.ipynb # end-to-end training notebook (Colab)
├── pyproject.toml         # dependencies (managed with uv)
├── .env.example           # required environment variables
└── .pre-commit-config.yaml
```

---

## Setup

**Prerequisites:** Python 3.12, [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/psaesha/LLM-Post-Training.git
cd LLM-Post-Training
uv sync
cp .env.example .env  # fill in the credentials
```

For Colab runs, open `notebooks/colab_runner.ipynb` — it mounts Drive, clones the repo, and calls `uv sync` automatically.

---

## Training Config (SFT)

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B` |
| Quantization | 4-bit NF4 (bfloat16 compute) |
| LoRA rank / alpha | 16 / 32 |
| LoRA target modules | q/k/v/o proj + gate/up/down proj |
| Trainable params | ~1.2% (~18M / 1.56B) |
| Dataset | `ai4bharat/indic-align` (Hindi) |
| Train / val size | 50,000 / 500 |
| Max sequence length | 1,024 |
| Epochs | 1 |
| Effective batch size | 16 (4 × 4 grad accumulation) |
| Learning rate | 2e-4 (cosine, 3% warmup) |
| Optimizer | `paged_adamw_8bit` |
| Precision | bf16 + gradient checkpointing |
| Logging | WandB |

Full config: [`configs/sft.yaml`](configs/sft.yaml)

---

## Evaluation

- **IndicGenBench** — automated benchmark for Indic language generation
- **Qualitative** — hand-graded prompt set covering factual Q&A, creative writing, code-mixed Hindi, and instruction following

---

## License

MIT — see [LICENSE](LICENSE).
