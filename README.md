# Task Two — Sentiment Analysis on Text Data

**Student:** Ernest Arthur  
**Task:** AI & Machine Learning Task 2  
**Dataset:** Twitter US Airline Sentiment (CrowdFlower/Figure Eight, 2015)  
**Goal:** Classify customer tweets as **negative**, **neutral**, or **positive**.

## Project summary

This project builds a three-class sentiment analysis model using **Python**, **Pandas**, **Scikit-learn**, **Matplotlib**, TF-IDF text features, and **Logistic Regression**.

The workflow:
1. Loads the labeled airline-tweet dataset.
2. Cleans text by lowercasing and removing URLs, user mentions, punctuation, and duplicate cleaned tweets.
3. Splits the data into 80% training and 20% testing using stratification and random state 42.
4. Converts text into TF-IDF unigram and bigram features with English stopword removal.
5. Trains a Logistic Regression classifier.
6. Evaluates accuracy, macro precision, macro recall, macro F1, a classification report, and a confusion matrix.

## Verified results

- **Cleaned samples:** 14,171
- **Training samples:** 11,336
- **Testing samples:** 2,835
- **Accuracy:** 76.72%
- **Macro precision:** 74.35%
- **Macro recall:** 62.35%
- **Macro F1:** 66.02%
- **Majority-class baseline:** 63.92%

See [EVALUATION.md](EVALUATION.md) for the full results, confusion matrix, limitations, and interpretation.

## Files

- `sentiment_analysis.py` — complete source code
- `requirements.txt` — Python dependencies
- `EVALUATION.md` — assignment evaluation and results
- `.gitignore` — excludes downloaded data and generated results
- `.github/workflows/task2-sentiment.yml` — automated training/evaluation workflow

## How to run

Install Python 3.10 or newer, then:

```bash
python -m pip install -r requirements.txt
python sentiment_analysis.py
```

On the first run, the script downloads `Tweets.csv` into `data/`. If automatic download fails, download the Twitter US Airline Sentiment dataset from Kaggle and place `Tweets.csv` in `data/`.

To use your own local copy:

```bash
python sentiment_analysis.py --data path/to/Tweets.csv
```

Generated outputs are saved in `results/`:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `sentiment_model.joblib`
- `example_predictions.txt`

## Dataset credit

CrowdFlower/Figure Eight, **Twitter US Airline Sentiment**:  
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

The third-party dataset is downloaded separately and is not committed to this repository.
