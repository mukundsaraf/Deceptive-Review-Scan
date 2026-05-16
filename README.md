# Deceptive-Review-Scan

An automated, scalable framework designed to detect fake and incentivized reviews on e-commerce platforms using Semi-Supervised Domain Adaptation. 

Addressing the World Economic Forum’s ranking of "Misinformation & Disinformation" as a top global risk, this repository tackles the economic threat of review fraud. Traditional models suffer from "vocabulary overfit" across differing product categories or require complete reviewer histories to work. This project implements a novel **Binary + Uncertain pipeline** using **XGBoost**, utilizing high-confidence pseudo-labels to specialize across 14 distinct target domains with zero manual annotation cost.

---

## 🚀 Key Features

* **Principled Uncertainty Zone:** Instead of forcing ambiguous reviews into a definitive category, the model introduces a three-tier decision state ($P < 0.35$ is **Real**, $P > 0.65$ is **Fake**, and $0.35 - 0.65$ is **Uncertain**).
* **Semi-Supervised Domain Adaptation:** Adapts a robust baseline classifier to unseen product spaces by utilizing high-confidence predictions ($\ge 0.70$) as training signals.
* **Substantial Uncertainty Reductions:** Achieves an average drop of ~20% in the uncertainty rate across target domains, rendering the classification system highly decisive.
* **Real-time Deployability:** Built using stylized text features and metadata metrics processed over XGBoost, allowing rapid, real-time inferencing.

---

## 📊 Pipeline Architecture

1. **Foundation:** An XGBoost base classifier is trained on a high-fidelity ground-truth dataset of 457,345 rows compiled via direct tracking of recruitment loops.
2. **Pseudo-Labeling:** The foundational model evaluates completely unlabeled category-specific datasets. Predictions yielding confidence scores $\ge 0.70$ are captured as pseudo-labels.
3. **Domain Specialization:** Domain-specific classifiers are fine-tuned using combined source data and pseudo-labeled target entries, reducing category ambiguity.

---

## 📈 Evaluation Results

By moving past standard ternary modeling (which trains a model on its own circular definitions) and deploying specialized domain agents, our models registered significant drops in uncertainty rates across 14 target categories:

| # | Domain Name | Uncertain Rate (Base) | Uncertain Rate (Adapted) | % Improvement |
|---|---|---|---|---|
| 1 | **Office** | 0.473 | 0.193 | **28.0%** |
| 2 | **Video Games** | 0.500 | 0.220 | **28.0%** |
| 3 | **Automotive** | 0.500 | 0.210 | **29.0%** |
| 4 | **Appliances** | 0.469 | 0.182 | **28.7%** |
| 5 | **Art** | 0.474 | 0.198 | **27.6%** |
| 6 | **Food** | 0.455 | 0.189 | **26.6%** |
| 7 | **Musical** | 0.496 | 0.270 | **22.6%** |
| 8 | **Industry** | 0.518 | 0.308 | **21.0%** |
| 9 | **Cell Phone** | 0.433 | 0.174 | **25.9%** |
| 10 | **Software** | 0.365 | 0.099 | **26.6%** |
| 11 | **Luxury** | 0.448 | 0.206 | **24.2%** |
| 12 | **Fashion** | 0.348 | 0.136 | **21.2%** |
| 13 | **Beauty** | 0.419 | 0.201 | **21.8%** |
| 14 | **Digital Music** | 0.633 | 0.241 | **39.2%** |

---

## 📁 Repository Structure

* `convert_json_to_domain_csv.py`: Utility script formatting massive structural JSON records into tabular cross-domain training components.
* `data_processing.py`: Extracted workflow scaling raw input matrices down into 13 high-signal features. Handles $log(1+x)$ transformations on community helpful markers and isolates structural validation indicators.
* `eda_reviews.py`: Exploratory parsing and data metrics profiling.
* `review_classifier_pipeline_binary.py`: Core foundational scripts establishing uncertainty bounds and running base model iterations.
* `visualise_results.py`: Graphing and validation tools computing drift dynamics and metric tables.
* `*_model.zip`: Serialized weights and fine-tuned configurations for the specialized e-commerce target categories.

---

## 🛠️ Data Strategy & Feature Processing

To preserve high performance without relying on text content memorization, we drop specific IDs, direct item URLs, and hardcoded markers. The initial 23 raw attributes are consolidated to 13 curated features:
* **Temporal Patterns:** Absolute review dates are parsed directly into month and year components to extract season-specific fraudulent bursts.
* **Binary Indicators:** Visual attachment arrays (`has_photos`) and context recruitment tracking indicators are stored as structured flags.
* **Structural Missings:** Missing attribute listings are retained intentionally as structural flags rather than using artificial replacement values, as missing parameters indicate a definitive signal for fake accounts.

---

## 🔮 Future Development Roadmap

1. **Richer Semantic Mapping:** Transitioning basic text counting pipelines into dense semantic engines using pre-trained sentence representations like BERT.
2. **Graph-based Relational Models:** Introducing user-to-product bipartite relational topologies to map and flag distributed review syndicates operating inside overlapping windows.
3. **Temporal Burst Detection:** Harnessing recurrent layer structures (LSTM) to parse sudden overnight review spikes.
4. **Interactive Enterprise Dashboards:** Constructing active tracking layouts to help web platforms preview real-time uncertainty indices instantly.

---

## 👥 Contributors

* **Mukund Saraf**
* **Krrish Singhania**
* **Vansh Jain**

*Developed under the guidance of Prof. Siddharth for the Machine Learning & Pattern Recognition Course (2026).*
