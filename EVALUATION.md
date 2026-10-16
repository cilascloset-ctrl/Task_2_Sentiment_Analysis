# Task Two — Evaluation Report

**Student:** Ernest Arthur  
**Assignment:** Sentiment Analysis on Text Data  
**Dataset:** Twitter US Airline Sentiment, CrowdFlower/Figure Eight (2015)

## Aim and method

The objective is to classify customer tweets as **negative**, **neutral**, or **positive** using Python, Pandas, Scikit-learn and Matplotlib.

1. Read the labeled Twitter US Airline Sentiment dataset using the `text` and `airline_sentiment` columns.
2. Convert text to lowercase and remove URLs, user mentions, punctuation, empty cleaned text, and duplicate cleaned tweets.
3. Split the cleaned data into **80% training / 20% testing** with stratification and random state 42.
4. Convert the text to numerical **TF-IDF unigram and bigram features** with English stopword removal.
5. Train a **Logistic Regression** classifier and evaluate it on unseen test samples.

## Measured results

| Metric | Value |
|---|---:|
| Cleaned, nonduplicate samples | 14,171 |
| Training samples | 11,336 |
| Testing samples | 2,835 |
| **Test accuracy** | **76.72%** |
| Majority-class baseline accuracy | 63.92% |
| Macro precision | 74.35% |
| Macro recall | 62.35% |
| Macro F1-score | 66.02% |

### Per-class classification

| Class | Precision | Recall | F1 | Test samples |
|---|---:|---:|---:|---:|
| Negative | 0.78 | 0.95 | 0.86 | 1,812 |
| Neutral | 0.64 | 0.36 | 0.46 | 586 |
| Positive | 0.81 | 0.56 | 0.66 | 437 |

### Confusion matrix

Rows are actual classes; columns are predicted classes. Label order: **negative, neutral, positive**.

| Actual / predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 1,718 | 60 | 34 |
| Neutral | 349 | 212 | 25 |
| Positive | 135 | 57 | 245 |

The model improves on the majority-class baseline by about **12.8 percentage points**. It performs very strongly on negative tweets, while neutral recall is lower, so macro F1 and the confusion matrix are important alongside overall accuracy.

## Limitations and possible improvements

- **Class imbalance:** Negative tweets are much more common than neutral and positive tweets.
- **Domain and age:** The dataset is about US airline experiences from 2015.
- **Language complexity:** Sarcasm, slang, spelling mistakes, emojis and mixed opinions are difficult for TF-IDF features.
- **Simple model:** Logistic Regression does not understand deep contextual meaning. Future comparisons could include class weighting, Naive Bayes, linear SVM, or transformer-based models.

## Reproduce the project

From the repository root:

```bash
python -m pip install -r requirements.txt
python sentiment_analysis.py
```

A successful run writes the following into `results/`:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `example_predictions.txt`
- `sentiment_model.joblib`

## Source attribution

CrowdFlower/Figure Eight, *Twitter US Airline Sentiment*, Kaggle:  
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

The dataset is downloaded separately and is not redistributed in this repository.
