# Transformer & LoRA Implementation from Scratch 
### [🇯🇵 日本語](#日本語-japanese) | [🇺🇸 English](#english)

-----


## 🇯🇵 日本語 (Japanese) <a id="日本語-japanese"></a>

## 1\. 概要

本プロジェクトは、論文「*Attention Is All You Need*」および「*LoRA: Low-Rank Adaptation of Large Language Models*」の理解を深めるために、**Hugging Face等の高レベルライブラリに依存せず、PyTorchのみでスクラッチ実装を行った学習記録**です。

実務レベルの最適化よりも、「アルゴリズムの内部挙動（数式とコードの対応）を肌感覚で完全に理解すること」を最優先の目的として設計されています。

> **注記:** 本リポジトリは学習目的で作成されたものであり、実務でのパフォーマンスよりも**コードの可読性と数式との対応関係**を重視しています。実用的なファインチューニングについては、[6.関連プロジェクト]のリポジトリをご参照ください。

## 2\. 本プロジェクトの主な特徴

  * **Pure PyTorch Implementation**
    `nn.Transformer` モジュール等は使用せず、Encoder/DecoderやAttention層をゼロから定義しています。ブラックボックス化された処理を排除し、計算グラフの透明性を確保しています。
  * **LoRA (Low-Rank Adaptation) の自作**
    `peft` ライブラリを使わず、線形層への低ランク行列 ($W + A \times B$) の注入ロジックを実装しました。凍結した重みと学習可能アダプターの計算分離を明確に記述しています。
  * **ハイブリッドな Attention 実装**
    学習目的の「手動計算経路（数式通りの実装）」と、パフォーマンスを考慮した「`F.scaled_dot_product_attention` 利用経路」の両方を実装し、比較検証が可能な設計にしています。
  * **構成のモジュール化**
    可読性と拡張性を意識したディレクトリ構成を採用し、各コンポーネント（Attention, LoRA, FeedForward）の役割を明確化しています。

## 3\. セットアップ

### ステップ 1: リポジトリのクローンと移動

```bash
git clone https://github.com/nnn087/transformer-lora-scratch.git
cd transformer-lora-scratch
```

### ステップ 2: 依存関係のインストール

本実装は PyTorch に依存しています。環境に合わせてインストールしてください。

```bash
pip install torch numpy
```

### ステップ 3: ディレクトリ構成の確認

本スクリプトは以下のディレクトリ構造で構成されています。

```text
.
├── src/
│   ├── layers/
│   │   ├── attention.py  <-- Multi-Head Attention (手動計算と高速化の実装)
│   │   ├── lora.py       <-- LoRAレイヤー (Freeze済み重み + アダプター)
│   │   └── ...
│   ├── models/
│   │   └── transformer.py <-- モデル全体統合
│   └── utils/
├── train.py
└── README.md
```

## 4\. 使用方法

### モデルの初期化と実行

`Transformer` クラスを呼び出し、LoRAランクを指定することで自動的にアダプターが適用されます。

```python
import torch
from src.models.transformer import Transformer

# 1. モデルの初期化 (LoRAランク指定により自動で適用)
model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=512,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    lora_rank=8  # LoRAを注入
)

# 2. ダミーデータの入力 (Batch Size, Seq Len)
src = torch.randint(0, 5000, (1, 10))
tgt = torch.randint(0, 5000, (1, 10))

# 3. Forward pass
output = model(src, tgt)
print(f"Output Shape: {output.shape}") # torch.Size([1, 10, 5000])
```

## 5\. 実装の詳細と開発プロセス

### 主要コンポーネント

特に以下のファイルに、学習と実装の工夫が反映されています。

  * **`src/layers/lora.py`**
    既存の `nn.Linear` 層をラップし、低ランク行列 $A, B$ を注入するLoRAレイヤーの実装です。
  * **`src/layers/attention.py`**
    `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V` の数式計算フローを詳細にコメントし、論文との対応関係を記述しています。

### 開発プロセスについて

本実装にあたっては、論文の読解補助およびコーディングのペアプログラミングパートナーとして **生成AI（LLM）** を積極的に活用しました。

生成されたコードはあくまで参考とし、全ての行について原論文の数式と照合・デバッグを行うことで、ロジックの正当性を担保しています。「なぜその計算になるのか」を自分自身で**咀嚼した内容**をコメントとして記述しています。

## 6\. 関連プロジェクト

  * **[Llama 3.1 MPS Fine-tuning Repository](https://github.com/nnn087/llama31-mps-finetuning)**
    こちらは実用目的で作成した、Mac (MPS) 環境でのLlama 3.1ファインチューニング用リポジトリです。本スクリプトで得た知見が、こちらの最適化に応用されています。

-----

## 🇺🇸 English \<a id="english"\>\</a\>

## 1\. Overview

This repository contains a scratch implementation of the Transformer model based on the paper "*Attention Is All You Need*" and LoRA (Low-Rank Adaptation), implemented purely in **PyTorch**.

The project focuses on prioritizing a deep, intuitive understanding of the internal algorithms (mapping equations to code) over production-level optimization.

> **Note:** This repository is for educational and research purposes, emphasizing **code readability and equation correspondence** rather than raw performance. For practical fine-tuning, please refer to the repository in the [6.Related Projects] section.

## 2\. Key Features

  * **Pure PyTorch Implementation**
    Built the `Transformer`, `Encoder`, and `Decoder` classes from scratch without using `nn.Transformer`. This eliminates black-box processes and ensures transparency in the computation graph.
  * **Custom LoRA Implementation**
    Implemented the Low-Rank Adaptation logic ($W + A \times B$) manually without using the `peft` library. Clearly separates the computation of frozen weights and trainable adapters.
  * **Hybrid Attention Mechanism**
    Implemented both a "manual calculation path" (for learning math) and a "fast path" (using `F.scaled_dot_product_attention`) in the Attention layer to allow for comparison.
  * **Modular Design**
    Adopted a directory structure that prioritizes readability and extensibility, clarifying the role of each component (Attention, LoRA, FeedForward).

## 3\. Setup

### Step 1: Clone and Move

```bash
git clone https://github.com/nnn087/transformer-lora-scratch.git
cd transformer-lora-scratch
```

### Step 2: Install Dependencies

This implementation depends on PyTorch. Please install it according to your environment.

```bash
pip install torch numpy
```

### Step 3: Check Directory Structure

This script assumes the following directory structure:

```text
.
├── src/
│   ├── layers/
│   │   ├── attention.py  <-- Multi-Head Attention (Manual & Fast imp.)
│   │   ├── lora.py       <-- LoRA Layer implementation
│   │   └── ...
│   ├── models/
│   │   └── transformer.py <-- Integrated Model
│   └── utils/
├── train.py
└── README.md
```

## 4\. Usage

### Model Initialization and Execution

Initialize the `Transformer` class and specify the LoRA rank to automatically apply adapters.

```python
import torch
from src.models.transformer import Transformer

# 1. Initialize model with LoRA
model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=512,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    lora_rank=8  # Inject LoRA
)

# 2. Dummy Input (Batch Size, Seq Len)
src = torch.randint(0, 5000, (1, 10))
tgt = torch.randint(0, 5000, (1, 10))

# 3. Forward pass
output = model(src, tgt)
print(f"Output Shape: {output.shape}") # torch.Size([1, 10, 5000])
```

## 5\. Implementation Details & Process

### Core Components

The following files reflect specific learning and implementation efforts:

  * **`src/layers/lora.py`**
    Implementation of the LoRA layer logic wrapping existing `nn.Linear` layers, injecting low-rank matrices $A$ and $B$.
    
  * **`src/layers/attention.py`**
    Multi-Head Attention implementation with detailed comments mapping code to equations (e.g., `Attention(Q, K, V)` logic).

### Development Process

I used Generative AI as a "pair programming partner" to accelerate my learning.

Instead of simply copying code, I verified every line against the original papers to ensure I understood *why* the implementation works. I have added comments documenting the content that I have personally **digested and verified**.

## 6\. Related Projects

  * **[Llama 3.1 MPS Fine-tuning Repository](https://github.com/nnn087/llama31-mps-finetuning)**
    My practical repository for fine-tuning Llama 3.1 on Mac (MPS) environment. The insights gained from this scratch implementation have been applied to optimize the practical repository.

-----
