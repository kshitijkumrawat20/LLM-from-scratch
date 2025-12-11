# 🤖 Building GPT-2 from Scratch

A comprehensive, step-by-step implementation of GPT-2 architecture from the ground up, including tokenization, attention mechanisms, transformer blocks, and fine-tuning on instruction datasets.

## 📚 Overview

This project provides a complete educational journey through building a Large Language Model (LLM) from scratch. Starting with basic tokenization and progressing through advanced topics like instruction fine-tuning, each notebook builds upon the previous concepts to create a fully functional GPT-2 implementation compatible with OpenAI's pretrained weights.

## 🎯 Learning Path

### Phase 1: Foundation (Notebooks 1-4)
Building the fundamental components required for any language model.

#### 1️⃣ [Creating Tokenizer for LLM](1_creating-tokenizer-for-llm.ipynb)
- Introduction to tokenization concepts
- Building a simple character-level tokenizer
- Understanding vocabulary and token IDs

#### 2️⃣ [BPE and DataLoaders](2_BPE_and_Dataloaders.ipynb)
- Byte Pair Encoding (BPE) algorithm
- GPT-2 tokenizer implementation using `tiktoken`
- Creating PyTorch DataLoaders for efficient batch processing

#### 3️⃣ [Token Embeddings](3_Token_Embeddings.ipynb)
- Converting tokens to dense vector representations
- Understanding embedding dimensions
- PyTorch `nn.Embedding` layer implementation

#### 4️⃣ [Positional Encoding](4_Positional_encoding.ipynb)
- Importance of position information in transformers
- Learned positional embeddings
- Combining token and position embeddings

### Phase 2: Attention Mechanisms (Notebooks 5-7)
Deep dive into the core innovation of transformers.

#### 5️⃣ [Self-Attention](5_SelfAttention.ipynb)
- Query, Key, Value (QKV) concept
- Scaled dot-product attention
- Attention score computation

#### 6️⃣ [Causal Attention](6_CausalAttention.ipynb)
- Masking future tokens for autoregressive generation
- Implementing causal masks
- Understanding attention flow in decoder-only models

#### 7️⃣ [Multi-Head Attention](7_multiHeatAttention.ipynb)
- Splitting attention into multiple heads
- Parallel attention computation
- Combining multi-head outputs

### Phase 3: Building Blocks (Notebooks 8-9)
Assembling the core transformer architecture.

#### 8️⃣ [Layer Normalization](8_layer_normalization.ipynb)
- Normalization techniques in deep learning
- Layer normalization implementation
- Pre-norm vs post-norm architectures

#### 9️⃣ [Coding Transformer Block](9_CodingTransformerBlock.ipynb)
- Complete transformer block implementation
- Multi-head attention + Feed-forward network
- Residual connections and dropout

### Phase 4: Complete Model (Notebooks 10-14)
Building and loading the full GPT-2 model.

#### 🔟 [Coding the GPT-2](10_Coding_the_GPT_2.ipynb)
- Complete GPT-2 architecture
- Stacking transformer blocks
- Output projection layer

#### 1️⃣1️⃣ [Calculating Loss](11_Calculating_loss.ipynb)
- Cross-entropy loss for language modeling
- Batch loss computation
- Evaluation metrics

#### 1️⃣2️⃣ [Coding Pretraining Loop](12_CodingPretrainingLoop.ipynb)
- Training loop implementation
- Gradient accumulation
- Checkpointing and evaluation

#### 1️⃣3️⃣ [Temperature Scaling and Top-K](13_TemperatureScalingAndTopK.ipynb)
- Text generation strategies
- Temperature parameter for controlling randomness
- Top-K and Top-P sampling

#### 1️⃣4️⃣ [Loading GPT-2 Weights](14_loading_gpt_2_weights_on_our_pytorch_model.ipynb)
- Downloading OpenAI's pretrained weights
- Weight mapping and loading
- Validating model outputs

### Phase 5: Fine-Tuning (Notebooks 15-16)
Adapting pretrained models for specific tasks.

#### 1️⃣5️⃣ [Fine-Tuning for Classification](15_Finetuning_for_classification.ipynb)
- Task-specific fine-tuning
- Classification head implementation
- Training and evaluation

#### 1️⃣6️⃣ [Instruction Fine-Tuning](16_instruction_fintuning.ipynb) ⭐
- **Stanford Alpaca dataset** (52,002 instruction-response pairs)
- Instruction prompt formatting
- Memory optimization strategies for large models
- Training GPT-2 Medium (355M parameters) with limited GPU memory
- Gradient accumulation techniques
- Model evaluation and response generation

## 🏗️ Model Architectures

This implementation supports all GPT-2 model sizes:

| Model | Parameters | Layers | Embedding Dim | Attention Heads |
|-------|-----------|--------|---------------|-----------------|
| **GPT-2 Small** | 124M | 12 | 768 | 12 |
| **GPT-2 Medium** | 355M | 24 | 1024 | 16 |
| **GPT-2 Large** | 774M | 36 | 1280 | 20 |
| **GPT-2 XL** | 1558M | 48 | 1600 | 25 |

## 💾 Dataset

### The Verdict (Pre-training)
- Sample text file for initial training experiments
- Located in: `the-verdict.txt`

### Stanford Alpaca (Instruction Fine-Tuning)
- **52,002 instruction-following examples**
- Generated from OpenAI's `text-davinci-003`
- Format: `instruction` → `input` (optional) → `output`
- Downloaded automatically in notebook 16
- Located in: `alpaca_data.json`

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.13+ recommended
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install tiktoken
pip install numpy pandas matplotlib
pip install tensorflow  # For loading TensorFlow checkpoints
pip install requests tqdm
pip install jupyter ipykernel
