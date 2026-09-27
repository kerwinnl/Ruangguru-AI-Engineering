# AI Bootcamp Portfolio

> A hands-on progression through data analysis, classical machine learning, deep learning, computer vision, generative AI, sequence modeling, information retrieval, and AI application development.

This repository contains my completed projects from an AI Engineering / Machine Learning bootcamp. The work progresses from foundational data workflows to modern transformer-based and generative AI systems.

## Project Overview

| # | Project | Main Focus | Technologies |
|---|---|---|---|
| 01 | [FIFA 21 Data Analysis](01-fifa21_project.ipynb) | Data cleaning, EDA, visualization | Python, Pandas, NumPy, Matplotlib |
| 02 | [FIFA 21 Machine Learning](02-machine_learning_project.ipynb) | Supervised classification | Scikit-learn, Logistic Regression |
| 03 | [Deep Learning Fundamentals](03-deep_learning_project.ipynb) | Neural networks, forward/backpropagation | NumPy, PyTorch |
| 04 | [PyTorch Deep Learning](04-pytorch_project.ipynb) | End-to-end neural network training | PyTorch, Scikit-learn, Plotly |
| 05 | [Image Classification Pipeline + Gradio](05-pipeline-and-gradio.ipynb) | Model inference and deployment | Hugging Face, ViT, Gradio |
| 06 | [Transfer Learning](06-transfer-learning.ipynb) | Vision Transformer fine-tuning | Transformers, Datasets, Evaluate, Gradio |
| 07 | [Database & Search](07-database_project.ipynb) | SQL and search systems | SQLite, Elasticsearch |
| 08 | [CNN / Image Classification](08-cnn-project.ipynb) | Image classification and model publishing | PyTorch, Hugging Face, ViT |
| 09 | [Stable Diffusion + ControlNet](09-stable-diffusion.ipynb) | Generative AI and controlled generation | Diffusers, Stable Diffusion, ControlNet |
| 10 | [RNN Encoder](10a-rnn-encoder.ipynb) | Sequence encoding and hidden states | RNN, Tanh |
| 11 | [RNN Decoder](10b-rnn-decoder.ipynb) | Sequence decoding and next-token prediction | RNN, embeddings |
| 12 | [LangChain RAG](11-langchain.ipynb) | Retrieval-augmented question answering | LangChain, FAISS, embeddings, LLMs, Gradio |
| 13 | [AI Portfolio / Final Project](12-ai-portfolio.ipynb) | End-to-end AI application | PyTorch, Transformers, Hugging Face |

## What I Built

### 01 — FIFA 21 Players: Data Analysis & Visualization

A complete exploratory data workflow using the FIFA 21 player dataset.

**Work covered**
- Removed redundant columns and formatting artifacts
- Cleaned newline and star (`★`) characters
- Handled missing values
- Converted `Value`, `Wage`, and `Release Clause` into numeric financial values
- Analyzed preferred-foot distribution
- Visualized the highest-rated players
- Compared teams using top-player performance
- Examined player value versus wage relationships

**Stack:** `Python` · `Pandas` · `NumPy` · `Matplotlib`

---

### 02 — FIFA 21 Supervised Machine Learning

A binary classification problem designed to identify "valuable" players from FIFA 21 attributes.

**Target**
- `Overall >= 75` → valuable player
- `Overall < 75` → non-valuable player

The model deliberately excludes `Overall` from the feature set to avoid target leakage.

**Model**
- Logistic Regression
- Train/test split
- Imputation
- Feature scaling
- Accuracy and F1 evaluation

**Observed notebook result**
- Accuracy: **96.43%**
- F1 Score: **86.35%**

**Stack:** `Scikit-learn` · `Pandas` · `NumPy`

---

### 03 — Deep Learning Fundamentals

A fundamentals-focused project covering the mechanics behind neural networks rather than treating them as black boxes.

**Topics**
- Matrix operations and neural-network forward passes
- Activation functions
- Network calculations
- Backpropagation
- Gradient-based optimization
- Manual cost calculation

The project also includes an optional two-layer network exercise.

**Stack:** `NumPy` · `PyTorch`

---

### 04 — PyTorch Deep Learning

An end-to-end deep learning workflow implemented directly with PyTorch.

The project uses a labeled classification dataset and builds a neural network with:
- Linear layers
- ReLU activation
- Cross-entropy loss
- Adam optimization
- Mini-batch training
- Validation-loss monitoring

**Model configuration**
- Hidden dimension: `64`
- Output classes: `2`
- Learning rate: `0.001`
- Epochs: `10`
- Batch size: `16`

**Observed notebook result**
- Test Accuracy: **93.86%**
- Validation Loss: **0.1649**

**Stack:** `PyTorch` · `Scikit-learn` · `Plotly`

---

### 05 — Image Classification Pipeline + Gradio

An inference pipeline built around a pretrained Hugging Face image classifier.

**Application**
- Accept an image URL
- Load and preprocess the image
- Run age-group classification
- Return prediction probabilities
- Wrap the model in a Gradio interface for interactive use

**Model**
- `nateraw/vit-age-classifier`

**Stack:** `Hugging Face Transformers` · `Vision Transformer (ViT)` · `Gradio` · `PIL`

---

### 06 — Transfer Learning for Emotion Recognition

A Vision Transformer transfer-learning workflow.

**Pipeline**
1. Load `FastJobs/Visual_Emotional_Analysis`
2. Explore the image dataset
3. Apply image preprocessing and augmentation
4. Start from a pretrained ViT
5. Fine-tune the classifier
6. Evaluate the model
7. Publish the trained model to Hugging Face
8. Build an interactive Gradio application combining emotion and age classification

**Base model**
- `google/vit-base-patch16-224-in21k`

**Stack:** `Transformers` · `Datasets` · `Evaluate` · `PyTorch` · `Hugging Face Hub` · `Gradio`

---

### 07 — Database & Search Engineering

A practical introduction to storing, indexing, and querying structured data through two different systems.

**SQL**
- SQLite database creation
- Table creation and data insertion
- Database verification
- SQL querying

**Search**
- Elasticsearch index creation
- Document ingestion
- Index verification
- Elasticsearch queries

**Stack:** `SQLite` · `SQL` · `Elasticsearch` · `Python`

---

### 08 — CNN / Image Classification

An image-classification project using the Hugging Face ecosystem.

The notebook works with the **Matthijs/snacks** dataset and demonstrates:
- Dataset loading
- Image preprocessing
- Label mapping
- Training/validation preparation
- Vision Transformer-based image classification
- Model evaluation
- Inference with confidence scores
- Model publishing to Hugging Face

**Base model**
- `google/vit-base-patch16-224-in21k`

**Published inference model**
- `silvering/vit-snack-classification-fp16`

**Stack:** `PyTorch` · `Transformers` · `Datasets` · `Evaluate` · `Hugging Face Hub`

---

### 09 — Stable Diffusion + ControlNet

A generative-AI project covering both fine-tuning and controlled image generation.

#### Fine-tuning
- Stable Diffusion pipeline setup
- Image dataset preparation
- Fine-tuning workflow
- Image generation and visualization

#### ControlNet — Canny Edge
- Edge extraction with Canny
- Conditioning Stable Diffusion on structural information
- Controlled image generation

#### ControlNet — Pose
- Human-pose extraction with OpenPose
- Pose-conditioned generation
- Stable Diffusion 2.1 ControlNet pipeline
- Memory/performance optimization

**Stack:** `Diffusers` · `Stable Diffusion` · `ControlNet` · `OpenPose` · `PyTorch` · `OpenCV`

---

### 10 — RNN Encoder

A manual sequence-modeling exercise designed to make the internal state of an RNN explicit.

**Concepts**
- One-hot word representations
- Feature encoding
- Hidden-state propagation
- Tanh activation
- Timestep-by-timestep computation

Example sequences include:
- `aku suka coklat`
- `aku tidak suka apel`

The second exercise extends the representation with a new `fruits` feature and recomputes the hidden states.

---

### 11 — RNN Decoder

A complementary sequence-decoding exercise using the hidden state produced by the encoder.

**Workflow**
- Transfer the encoder's final hidden state to the decoder
- Select the most probable output token
- Feed the predicted token into the next timestep
- Update embeddings and hidden states
- Continue until the end-of-sequence token

This demonstrates the basic mechanics behind autoregressive sequence generation and encoder-decoder architectures.

---

### 12 — LangChain RAG Application

A retrieval-augmented question-answering system over *Pride and Prejudice*.

**Pipeline**

```text
Book Text
   ↓
Document Loading
   ↓
Text Splitting
   ↓
Sentence-Transformer Embeddings
   ↓
FAISS Vector Store
   ↓
Retriever
   ↓
LLM
   ↓
Question Answer
```

**Components**
- `RecursiveCharacterTextSplitter`
- Hugging Face `all-MiniLM-L6-v2` embeddings
- FAISS vector database
- LangChain `RetrievalQA`
- LLM-backed answer generation
- Gradio interface

**Stack:** `LangChain` · `FAISS` · `Sentence Transformers` · `LLM` · `Gradio`

---

### 13 — AI Portfolio / Final Project

The final project consolidates the computer-vision and deep-learning skills developed throughout the bootcamp.

The notebook includes:
- Image dataset loading
- Train/validation/test splitting
- A CNN architecture implemented in PyTorch
- Convolution, batch normalization, pooling, and fully connected layers
- Cross-entropy loss
- Adam optimization
- Training and validation-loss tracking
- Accuracy evaluation

It also explores a Hugging Face Vision Transformer workflow on a plant-disease image dataset, including preprocessing, fine-tuning, evaluation, inference, and model publication.

**Stack:** `PyTorch` · `Torchvision` · `Transformers` · `Hugging Face Datasets` · `Hugging Face Hub`

---

## Technical Skills Demonstrated

### Machine Learning
- Supervised classification
- Feature engineering and selection
- Data imputation
- Feature scaling
- Logistic Regression
- Model evaluation
- Accuracy and F1 score

### Deep Learning
- Forward propagation
- Backpropagation
- Gradient descent
- Neural-network architecture design
- PyTorch training loops
- Loss optimization
- Validation and test evaluation

### Computer Vision
- CNNs
- Vision Transformers
- Transfer learning
- Image preprocessing and augmentation
- Image classification
- Model inference
- Confidence scoring

### Generative AI
- Stable Diffusion
- Fine-tuning
- ControlNet
- Canny-edge conditioning
- OpenPose conditioning
- Diffusion pipelines

### NLP & LLM Systems
- RNN encoder-decoder architectures
- One-hot encoding
- Hidden-state propagation
- Embeddings
- Vector search
- Retrieval-Augmented Generation
- LangChain
- LLM-based question answering

### Data & Search
- Pandas
- SQL
- SQLite
- Elasticsearch
- FAISS

### Deployment & MLOps Foundations
- Hugging Face Model Hub
- Interactive Gradio applications
- Model inference pipelines
- Dataset/model publishing

---

## Repository Structure

```text
.
├── 01-fifa21_project.ipynb
├── 02-machine_learning_project.ipynb
├── 03-deep_learning_project.ipynb
├── 04-pytorch_project.ipynb
├── 05-pipeline-and-gradio.ipynb
├── 06-transfer-learning.ipynb
├── 07-database_project.ipynb
├── 08-cnn-project.ipynb
├── 09-stable-diffusion.ipynb
├── 10a-rnn-encoder.ipynb
├── 10b-rnn-decoder.ipynb
├── 11-langchain.ipynb
├── 12-ai-portfolio.ipynb
├── Copy of RNN Encoder Project.xlsx
└── Copy of RNN Decoder Project.xlsx
```

## Learning Progression

```text
Data Analysis
      │
      ▼
Classical Machine Learning
      │
      ▼
Neural Network Fundamentals
      │
      ▼
PyTorch
      │
      ▼
Computer Vision
      │
      ├──────────────► Transfer Learning / ViT
      │
      └──────────────► CNNs
      │
      ▼
Generative AI
      │
      ├──────────────► Stable Diffusion
      └──────────────► ControlNet
      │
      ▼
Sequence Modeling
      │
      ├──────────────► RNN Encoder
      └──────────────► RNN Decoder
      │
      ▼
Retrieval + LLM Applications
      │
      ▼
End-to-End AI Portfolio
```

## Core Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square" />
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=flat-square" />
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=flat-square" />
  <img src="https://img.shields.io/badge/Elasticsearch-005571?style=flat-square&logo=elasticsearch&logoColor=white" />
</p>

## Notes

This repository is primarily a learning portfolio. Some notebooks are structured as guided bootcamp assignments, while others implement more complete end-to-end workflows.

Performance figures are reported only where they were explicitly present in the executed notebooks. For projects without recorded metrics, the README describes the implemented workflow rather than inventing benchmark results.

## Author

**Kerwin**

Physics undergraduate building toward work at the intersection of **AI, machine learning, scientific computing, and research**.

---

*Built through hands-on implementation, experimentation, and iterative learning.*
