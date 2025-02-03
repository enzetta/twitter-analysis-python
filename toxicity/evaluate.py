import json
from collections import defaultdict
from transformers import pipeline

# Define models for sentiment analysis and toxicity detection
TOXICITY_MODEL = "textdetox/xlmr-large-toxicity-classifier"
SENTIMENT_MODEL = "oliverguhr/german-sentiment-bert"

# Load the sentiment and toxicity classification pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
toxicity_pipeline = pipeline("text-classification", model=TOXICITY_MODEL)

# Load the dataset
with open("data/toxicity-evaluation-data.json", "r", encoding="utf-8") as file:
    tweets = json.load(file)

# Define possible labels
sentiment_labels = ["positive", "neutral", "negative"]
toxicity_labels = ["non-toxic", "neutral", "toxic"]

# Initialize confusion matrices
sentiment_confusion = defaultdict(lambda: defaultdict(int))
toxicity_confusion = defaultdict(lambda: defaultdict(int))

# Counters for evaluation
sentiment_correct = 0
toxicity_correct = 0

# Total count
total_sentiment = len(tweets)
total_toxicity = len(tweets)

# Evaluation
for tweet in tweets:
    text = tweet["text"]
    true_sentiment = tweet["sentiment"]
    true_toxicity = tweet["toxicity_label"]

    # Model predictions
    pred_sentiment = sentiment_pipeline(text)[0]["label"]
    pred_toxicity = toxicity_pipeline(text)[0]["label"]

    # Update confusion matrices
    sentiment_confusion[true_sentiment][pred_sentiment] += 1
    toxicity_confusion[true_toxicity][pred_toxicity] += 1

    # Accuracy count
    if true_sentiment == pred_sentiment:
        sentiment_correct += 1
    if true_toxicity == pred_toxicity:
        toxicity_correct += 1


# Function to print confusion matrix
def print_confusion_matrix(matrix, title, labels):
    print(f"\n{title} Confusion Matrix:")
    print(" " * 15 + " ".join(f"{label:>10}" for label in labels))
    for true_label in labels:
        row = f"{true_label:<15}" + " ".join(
            f"{matrix[true_label][pred_label]:>10}" for pred_label in labels)
        print(row)


# Function to calculate precision, recall, F1-score
def calculate_metrics(matrix, labels):
    metrics = {}

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[row][label] for row in labels
                 if row != label)  # False positives
        fn = sum(matrix[label][col] for col in labels
                 if col != label)  # False negatives
        tn = sum(matrix[row][col] for row in labels for col in labels
                 if row != label and col != label)  # True negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (
            precision + recall) > 0 else 0

        metrics[label] = {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1_score, 4),
        }

    return metrics


# Calculate and print results
sentiment_metrics = calculate_metrics(sentiment_confusion, sentiment_labels)
toxicity_metrics = calculate_metrics(toxicity_confusion, toxicity_labels)

# Print confusion matrices
print_confusion_matrix(sentiment_confusion, "Sentiment", sentiment_labels)
print_confusion_matrix(toxicity_confusion, "Toxicity", toxicity_labels)

# Print accuracy
sentiment_accuracy = sentiment_correct / total_sentiment
toxicity_accuracy = toxicity_correct / total_toxicity
print(f"\nSentiment Accuracy: {sentiment_accuracy:.4f}")
print(f"Toxicity Accuracy: {toxicity_accuracy:.4f}")

# Print precision, recall, and F1-score
print("\nSentiment Classification Metrics:")
for label, values in sentiment_metrics.items():
    print(f"{label}: {values}")

print("\nToxicity Classification Metrics:")
for label, values in toxicity_metrics.items():
    print(f"{label}: {values}")
