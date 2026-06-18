# BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer built from scratch in Python as part of my journey to understand the fundamental building blocks of modern AI systems and Large Language Models (LLMs).

## Motivation

I created this project to gain a deeper understanding of how text is processed before being fed into Machine Learning and Deep Learning models.

While learning about topics such as:

* Natural Language Processing (NLP)
* Embeddings
* Neural Networks
* Transformers
* Large Language Models (LLMs)
* Vector Representations
* Language Understanding

I realized that tokenization is one of the most important steps in the entire AI pipeline. Instead of using existing libraries, I decided to implement a BPE tokenizer from scratch to understand how modern language models convert raw text into tokens.

## What I Learned

Through this project, I explored:

* Text preprocessing techniques
* Vocabulary construction
* Frequency analysis
* Subword tokenization
* Byte Pair Encoding (BPE)
* Data structures and algorithms
* The relationship between tokens and embeddings
* How tokenized text is used by neural networks
* Foundations of transformer-based models

## Features

* Custom BPE implementation
* Token frequency analysis
* Pair frequency counting
* Iterative token merging
* Vocabulary generation
* File handling and persistence
* Modular Python architecture
* Error handling and validation

## Project Structure

```text
BPE-Tokenizer/
│
├── BPE_Engine.py
├── Frequency_counter.py
├── Error_handler.py
├── Save_files.py
├── main.py
└── README.md
```

## How BPE Works

1. Split text into basic symbols.
2. Count adjacent pair frequencies.
3. Find the most frequent pair.
4. Merge the pair into a new token.
5. Repeat until the target vocabulary size is reached.
6. Use the learned vocabulary to tokenize unseen text.

## Why This Project Matters

Modern AI systems do not directly understand words. They operate on tokens and numerical representations.

Building this tokenizer helped me understand the complete path:

```text
Raw Text
   ↓
Tokenization
   ↓
Vocabulary
   ↓
Embeddings
   ↓
Neural Networks
   ↓
Transformer Models
   ↓
Language Understanding
```

This project serves as a foundation for exploring more advanced topics such as embedding models, attention mechanisms, transformers, and large-scale language models.

## Technologies Used

* Python
* NLP Concepts
* Byte Pair Encoding (BPE)
* Data Structures
* Algorithm Design

## Future Improvements

* BPE Decoder
* Unicode Support
* Faster Training Pipeline
* Embedding Visualization
* Transformer Integration
* Dataset-Based Vocabulary Training

## About Me

Hi, I'm Jatin.

I am an aspiring Data Scientist and Machine Learning Engineer who enjoys learning by building projects from scratch. I use AI as a learning companion to explore complex topics and gain a deeper understanding of concepts such as tokenization, embeddings, neural networks, transformers, and modern language models.

My goal is not only to use AI tools but also to understand how they work internally and develop the skills required to build intelligent systems in the future.
