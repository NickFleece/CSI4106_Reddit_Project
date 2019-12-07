import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk

###
### This file tests our models on the tweet data at: "data/testing_tweets.csv" & "data/testing_tags.csv"
### It prints out the precision, recall, and accuracy
###

data = {
    "testing_tags": [],
    "testing_tweets": []
}

#parse the training and testing data we need and store it in the dictionary
for key in data.keys():
    with open(f"data/{key}.csv", encoding="utf-8") as csvFile:
        print(f"Parsing file: {key}")
        csv_reader = csv.reader(csvFile)
        header = True
        for row in csv_reader:
            if header:
                header = False
                continue
            data[key].append(row[1])

print("Vectorizing")
with open("models/vectorizer", 'rb') as file:
    count_vect = pickle.load(file)
test_counts = count_vect.transform(data["testing_tweets"])

models = {
    "svm": None,
    "mlp": None
}
#load our models using pickle
for i in models.keys():
    with open(f"models/{i}", 'rb') as pickle_file:
        models[i] = pickle.load(pickle_file)

predictions = {}
#use our models to predict
for model in models.keys():
    print(f"Predicting {model}")
    predictions[model] = models[model].predict(test_counts)
    true_positives = 1
    true_negatives = 1
    false_positives = 1
    false_negatives = 1
    count = 0
    print("Done predicting, parsing results...")
    for prediction, tag in zip(predictions[model], data["testing_tags"]):
        count += 1
        # 4 = positive
        # 0 = negative
        if prediction == "4" and tag == "4":
            true_positives += 1
        elif prediction == "0" and tag == "0":
            true_negatives += 1
        elif prediction == "4" and tag == "0":
            false_positives += 1
        elif prediction == "0" and tag == "4":
            false_negatives += 1
    print(f"{model}:\n-Precision: {true_positives / (true_positives + false_positives)}"
          f"\n-Recall: {true_positives / (true_positives + false_negatives)}"
          f"\n-Accuracy: {(true_positives + true_negatives) / count}")


###
### VADER
###
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('vader_lexicon')

true_positives = 1
true_negatives = 1
false_positives = 1
false_negatives = 1
count = 0

tags = ["pos", "neg"]
sia = SIA()
vader_results = []
for tweet, tag in zip(data["testing_tweets"], data["testing_tags"]):
    #get the vader score
    pol_score = sia.polarity_scores(tweet)
    max = "neu"
    for t in tags:
        if pol_score[t] > pol_score[max]:
            max = t
    if max == "neu":
        continue
    count += 1
    if max == "pos" and tag == "4":
        true_positives += 1
    elif max == "neg" and tag == "0":
        true_negatives += 1
    elif max == "pos" and tag == "0" or max == "neu":
        false_positives += 1
    elif max == "neg" and tag == "4" or max == "neu":
        false_negatives += 1

print(f"VADER:\n-Precision: {true_positives / (true_positives + false_positives)}"
          f"\n-Recall: {true_positives / (true_positives + false_negatives)}"
          f"\n-Accuracy: {(true_positives + true_negatives) / count}")