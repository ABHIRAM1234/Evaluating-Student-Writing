# Feedback Prize: Advanced NLP for Argumentative Writing Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for a state-of-the-art NLP solution for the [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021/overview) Kaggle competition. The project's goal is to automatically identify and classify seven different argumentative and rhetorical elements in student essays.

Our final model achieves an **F1 score of 0.74**, matching the performance of the 2nd place solution on the private leaderboard.

---

## 📌 Project Overview

Manually providing detailed feedback on student writing is a time-consuming process that doesn't scale. This project addresses that challenge by building an automated system to parse student essays and identify the distinct components of their arguments (e.g., `Lead`, `Position`, `Claim`, `Evidence`). Such a tool can serve as the backbone for educational software that provides instant, formative feedback, helping students master the structure of effective writing.

### Key Technical Highlights
*   **Multi-Stage Architecture:** A sophisticated ensemble of diverse models, not a single monolithic solution.
*   **Long-Document Transformers:** Utilizes `Longformer` and `BigBird` to effectively process essays longer than the standard 512-token limit.
*   **Hybrid Modeling:** Combines deep learning (transformers) with classic machine learning (XGBoost on learned embeddings) to maximize predictive diversity.
*   **Advanced Post-Processing:** Implements **Weighted Box Fusion (WBF)**, a novel technique borrowed from computer vision, to intelligently merge model predictions and boost accuracy.

---

## ⚙️ Technical Architecture

The solution is a multi-stage pipeline designed for maximum accuracy and robustness.

```mermaid
graph TD
    A[Raw Essay Text] --> B{Tokenization & BIO Tagging};
    B --> C[Longformer Model];
    B --> D[BigBird Model];
    
    subgraph Hybrid Model Path
        direction LR
        B --> E1[Get Embeddings];
        E1 --> E2[Feature Engineering];
        E2 --> E3[XGBoost Model];
    end

    C --> F[Ensemble Predictions];
    D --> F;
    E3 --> F;

    F --> G[Weighted Box Fusion (WBF)];
    G --> H[Final Labeled Discourse Elements];
```

1.  **Preprocessing:** Essays are tokenized and assigned BIO (Beginning, Inside, Outside) tags for each of the 7 discourse classes.
2.  **Parallel Modeling:** The tokenized input is fed into three diverse model types in parallel:
    *   A fine-tuned `Longformer` model.
    *   A fine-tuned `BigBird` model.
    *   An `XGBoost` model trained on features derived from the Longformer's hidden-state embeddings.
3.  **Ensembling & Fusion:** The predictions from all models are combined. Instead of a simple vote, we use **Weighted Box Fusion (WBF)** to merge the predicted text spans (treated as "bounding boxes") based on their confidence scores, producing a single, highly accurate set of final labels.

---

## 📊 Results

The final ensemble model was evaluated using the competition's official metric, a micro F1-score.

| Model | Public F1 Score | Private F1 Score |
| :--- | :--- | :--- |
| **Final Ensemble (with WBF)** | **~0.73** | **0.740** |
| Competition 2nd Place Solution | 0.727 | 0.740 |

Our model successfully replicates the state-of-the-art performance, demonstrating the effectiveness of the hybrid ensemble and the WBF post-processing pipeline.

---

## 🚀 How to Use

### Prerequisites
*   Python 3.8+
*   PyTorch
*   HuggingFace Transformers, Datasets
*   Scikit-learn, XGBoost
*   CUDA for GPU acceleration (recommended)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/feedback-prize-nlp.git
    cd feedback-prize-nlp
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the data from the [Kaggle competition page](https://www.kaggle.com/competitions/feedback-prize-2021/data) and place it in the `data/` directory.

### Training a Model
To train the full ensemble, run the training script with the appropriate configuration file:
```bash
python src/train.py --config configs/final_ensemble_config.yaml
```

### Running Inference
To make predictions on new text, use the inference script with a trained model checkpoint:
```bash
python src/predict.py --model_path models/final_model.bin --text_file path/to/your/essay.txt
```

---

## 📁 Directory Structure

```
.
├── configs/                # YAML configuration files for different models
├── data/                   # Raw and processed data (from Kaggle)
├── models/                 # Saved model checkpoints
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/                    # Source code for the project
│   ├── data_loader.py      # Data loading and preprocessing scripts
│   ├── engine.py           # Core training and evaluation loops
│   ├── model.py            # Model definitions (Longformer, BigBird, etc.)
│   ├── post_processing.py  # Weighted Box Fusion implementation
│   ├── train.py            # Main training script
│   └── predict.py          # Inference script
├── requirements.txt        # Project dependencies
└── README.md```

---

## 🙏 Acknowledgements

This project builds upon the fantastic public work shared by the Kaggle community. The techniques implemented here were inspired by the insights and code from several top-performing public notebooks, including:
*   [2nd Place Solution - CV741 Public727 Private740](https://www.kaggle.com/code/cdeotte/2nd-place-solution-cv741-public727-private740) by Chris Deotte
*   [XGB+LGB Feedback Prize CV 0.7322](https://www.kaggle.com/code/aerdem4/xgb-lgb-feedback-prize-cv-0-7322) by aerdem4
*   [feedback-two-stage-lb0.727](https://www.kaggle.com/code/wht1996/feedback-two-stage-lb0-727) by wht1996
*   [[Last Dance 2] FPrize WBF](https://www.kaggle.com/code/amedprof/last-dance-2-fprize-wbf) by amedprof
