<div align="center">

# CARE: Curriculum-based Adversarial Robust Enhancement for RAG

</div>

## 🌟 Introduction

This repository is the official implementation of **CARE** a novel framework designed to enhance the robustness of Retrieval-Augmented Generation (RAG) systems against attacks.


### Key Features

- 🔄 **Adversarial Reforcement Training**: Alternates between RL training and dynamic attack optimization
- ⚡ **TextGrad-based Attack Optimization**: Uses gradient-based text optimization to craft stronger adversarial documents
- 🛡️ **Enhanced Robustness**: Significantly improves model resilience to retrieval poisoning attacks

---

## 🔧 Installation

Clone this repository, then create a conda environment and install the required packages.

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/CARE.git
cd CARE

# Create conda environment
conda create -n care python=3.10
conda activate care

# Install packages
pip install -r requirements.txt
```

> 💡 **Note:** If you encounter any issues when installing the Python packages, we recommend following the official installation instructions provided by [FlashRAG#Installation](https://github.com/RUC-NLPIR/FlashRAG/tree/main#wrench-installation).

If you are using Docker, run the command below:

```bash
# pull vllm docker image
docker pull vllm/vllm-openai:v0.10.2

# And install Additional Package
pip install textgrad
pip install falshrag
```

---

## 📊 Dataset

We conduct our training and evaluation across multiple multi-hop QA benchmarks:

| Type | Dataset | Train Num | Test Num  |
|------|---------|----------|-----------|
| Multi-hop QA | HotpotQA | 90447 | 7405 |
| Multi-hop QA | 2WikiMultiHopQA | 113284 | 7405 |
| Multi-hop QA | MuSiQue | 19938 | 2417 |
| Multi-hop QA | Bamboogle | Null | 125 |

### Data Collection

For initial training data collection (multi-hop QA with retrieved documents), please refer to the [DRAG repository](https://github.com/Huenao/Debate-Augmented-RAG) which provides the data collection pipeline.


The dataset using in our experiment are provided [here](https://huggingface.co/datasets/An998/CARE_data).



---

## 📚 Document Corpus & Index

We use the `wiki18_100w` dataset as the document corpus. Both the document corpus and the index can be downloaded from:

- [ModelScope Dataset](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files) (look in `retrieval_corpus` folder)

---

## 🤖 Models

### Retriever Model

We use `e5-base-v2` as the default retriever. Download from: [🤗 intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)

### Generator Models

This project supports all LLMs compatible with HuggingFace and vLLM. Recommended models:

| Model | Link |
|-------|------|
| Qwen3-4B-Instruct | [🔗 Link](https://huggingface.co/Qwen/Qwen3-4B-Instruct) |
| Qwen3-30B-A3B-Instruct | [🔗 Link](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct) |
| Llama-3.1-8B-Instruct | [🔗 Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |

---

## 🏋️ Training Pipeline

CARE follows an iterative curriculum-based adversarial training pipeline:

### Step 1: Data Collection

Collect multi-hop QA training data with retrieved documents using the DRAG framework:

```bash
# Follow DRAG repository for data collection
# https://github.com/Huenao/Debate-Augmented-RAG
```

### Step 2: Difficulty Measurement

Measure the difficulty of each training sample using sampling-based evaluation:

```bash
python src/difficulty_measurement.py \
    --model_path /path/to/base_model \
    --input_file data/train_data/pure_data/pure_hotpotqa_train.json \
    --output_file data/train_data/pure_data/pure_hotpotqa_train_with_difficulty.json \
    --num_samples 20 \
    --batch_size 8
```

### Step 3: Initial Attack Generation

Generate initial adversarial documents with wrong answer injection:

```bash
python src/gen_ini_attack.py \
    --model_path /path/to/attack_generator_model \
    --input_file data/train_data/pure_data/pure_hotpotqa_train_with_difficulty.json \
    --output_file data/train_data/attack_data/attack_round1_train_rl_prompt.json
```

### Step 4-6: Iterative RL Training

We use [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for reinforcement learning. Run the training script:

```bash
# Round 1 Training
bash train.sh  # Configure with attack_round1_train_rl_prompt.json
```

After each round, optimize attacks using TextGrad:

```bash
# Dynamic Attack Optimization (after round N, before round N+1)
python src/dynamic_attack.py \
    --model_path /path/to/trained_model_roundN \
    --input_file data/train_data/pure_data/pure_data_for_round.json \
    --output_file data/train_data/attack_data/attack_roundN+1_train_rl_prompt.json
```

Then continue training with the new adversarial data:

```bash
# Round N+1 Training
bash train.sh  # Configure with attack_roundN+1_train_rl_prompt.json
```


## 📈 Evaluation

### Running Evaluation

Evaluate CARE-trained models on poisoned/counterfactual test sets:

```bash
python src/answer_vllm_pipeline.py \
    --model_path /path/to/care_trained_model \
    --test_file data/test_data/Poisoned/poisoned_hotpotqa_test.jsonl \
    --output_file results/poisoned_hotpotqa_results.json \
    --tensor_parallel_size 2
```

### Evaluation Metrics

- **Exact Match (EM)**: Strict matching after normalization
- **F1 Score**: Token-level F1 between prediction and ground truth

---

## 📁 Project Structure

```bash
CARE/
├── data/
│   ├── train_data/
│   │   ├── pure_data/              # Clean training data
│   │   └── attack_data/            # Adversarial training data
│   └── test_data/
│       ├── Poisoned/               # Poisoned test sets
│       └── counterfact/            # Counterfactual test sets
├── src/
│   ├── difficulty_measurement.py   # Sample difficulty evaluation
│   ├── gen_ini_attack.py           # Initial attack document generation
│   ├── dynamic_attack.py           # TextGrad-based attack optimization
│   └── answer_vllm_pipeline.py     # vLLM-based evaluation pipeline
├── logs/                           # Training logs
├── output/                         # Model outputs
├── results/                        # Evaluation results
├── train.sh                        # RL training script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```


---

## ✨ Acknowledgments

We gratefully acknowledge the following projects:

- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG): A Python toolkit for the reproduction and development of Retrieval Augmented Generation (RAG) research.
- [DRAG](https://github.com/Huenao/Debate-Augmented-RAG): Debate-Augmented RAG framework for training data collection.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): An Easy-to-use, Scalable and High-performance RLHF Framework.
- [TextGrad](https://github.com/zou-group/textgrad): Automatic "Differentiation" via Text for gradient-based text optimization.

---
