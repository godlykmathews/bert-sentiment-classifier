# BERT Sentiment Classifier

A Streamlit app that classifies reviews as **Negative**, **Neutral**, or **Positive** using a fine-tuned BERT model. It displays the predicted sentiment and probabilities for all three classes.

## Setup

The app was checked with Python 3.14 and the package versions in `requirements.txt`.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` contains only the app's dependencies. The training notebook installs its own dependencies in its first cell.

## Model files

Hugging Face model: [godlykmathews/BERT-Sentiment_Classifier](https://huggingface.co/godlykmathews/BERT-Sentiment_Classifier).

Run the model-saving cell in [notebooks/train_bert.ipynb](notebooks/train_bert.ipynb). If training in Colab, download the entire saved model directory and extract it locally.

The notebook saves a folder named `bert_sentiment_model`. **Rename it to `bert-sentiment-model`** to match the app's current model path, then place it beside `app.py`:

```text
bert-sentimentclassifier/
├── app.py
├── requirements.txt
├── notebooks/
│   └── train_bert.ipynb
└── bert-sentiment-model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

The app also accepts `notebooks/bert-sentiment-model/`. Keep all files produced by the model-saving cell. The app loads local files and does not download a model automatically.

Model directories and training outputs are ignored by Git. Model files already committed remain tracked; `.gitignore` does not remove them from Git history.

## Run

From the project directory, with the virtual environment activated:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, enter a review, and click **Analyze**. The app uses CPU inference and analyzes up to 128 tokens per review.

## Training

The notebook fine-tunes [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) with PyTorch and Hugging Face Transformers on [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full).

| Yelp rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

Three-star reviews serve as a neutral proxy and can contain mixed opinions.

The notebook uses 15,000 training reviews, 1,500 validation reviews, and 3,000 test reviews. Training and validation come from the official training split; test reviews come from the official test split. It trains for three epochs and selects the checkpoint with the best validation macro F1.

## Results

Results recorded in the executed notebook on the held-out test sample:

| Metric | Value |
|---|---:|
| Accuracy | 77.40% |
| Macro precision | 0.7281 |
| Macro recall | 0.7275 |
| Macro F1 | 0.7278 |

The notebook includes the per-class metrics and confusion matrix. Neutral was the weakest class, with an F1 of 0.4996. These results describe the Yelp test sample; class probabilities are not calibrated guarantees of correctness.
