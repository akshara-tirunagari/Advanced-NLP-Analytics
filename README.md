# Advanced NLP & Unstructured Data Analytics 🧠

Advanced Natural Language Processing portfolio featuring LLM benchmarking (Claude/Llama), BERTopic modeling, and Deep Learning architectures (LSTM/RNN) on Yelp data

This repository contains a collection of advanced Natural Language Processing (NLP) projects completed as part of the MS Business Analytics program at Arizona State University. The projects utilize the Yelp Open Dataset to explore sentiment analysis, topic modeling, and text classification using modern architectures.

## 📊 Data Availability
**Note:** The primary dataset (`restaurant_reviews_az.csv`) is not included in this repository due to GitHub's file size constraints (>25MB).

* **Source:** The data is a processed subset of the [Yelp Open Dataset](https://www.yelp.com/dataset).
* **Scope:** It specifically filters for restaurant reviews in Arizona, containing text data, star ratings, and sentiment labels used for training and validation across the projects below.

## 📂 Project Overview

### 1. Generative AI & LLM Benchmarking (`LLM_Sentiment_Analysis_FewShot.ipynb`)
**Tech Stack:** Python, Anthropic (Claude 3 Sonnet), LLaMA, Prompt Engineering.
* **Objective:** Evaluated the efficacy of Large Language Models (LLMs) for sentiment classification without model fine-tuning.
* **Methodology:** Implemented **Zero-Shot** and **Few-Shot** prompting strategies to classify reviews.
* **Outcome:** Compared performance metrics (Precision, Recall, F1-Score) across models, identifying where Few-Shot prompting significantly reduced hallucination and improved accuracy compared to Zero-Shot baselines.

### 2. Topic Modeling with BERTopic (`Topic_Modeling_BERTopic.ipynb`)
**Tech Stack:** BERTopic, UMAP, BERT (Transformer embeddings), Pandas.
* **Objective:** Extracted latent themes from thousands of unstructured customer reviews to identify business insights.
* **Methodology:** Used **UMAP** for dimensionality reduction and **c-TF-IDF** to create dense clusters of topics.
* **Key Findings:** Visualized topic hierarchies and temporal trends (2020-2021), revealing how customer complaints shifted from "service speed" to "sanitation/masks" during specific periods.

### 3. Deep Learning Text Classification (`Deep_Learning_Sentiment_Comparison.ipynb`)
**Tech Stack:** TensorFlow, Keras, Scikit-Learn, GloVe Embeddings.
* **Objective:** Built and compared neural network architectures for sentiment quantification.
* **Models Built:**
    * **ANN (Artificial Neural Network):** Baseline model using TF-IDF vectors.
    * **RNN (Recurrent Neural Network):** Captured sequential dependencies in text.
    * **LSTM (Long Short-Term Memory):** Solved the vanishing gradient problem to better understand long context reviews.
* **Result:** The LSTM model achieved the highest validation accuracy, demonstrating superior handling of context in longer reviews compared to simple ANNs.

## 🛠️ Installation & Usage

To replicate the analysis, ensure you have the required packages:

```bash
pip install tensorflow bertopic anthropic pandas scikit-learn
