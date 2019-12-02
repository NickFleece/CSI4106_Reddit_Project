import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

data = {
    "training_tags": [],
    "testing_tags": [],
    "training_tweets": [],
    "testing_tweets": []
}

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

# def no_alpha_no_stopwords(tweets):
#     new_tweets = []
#     for tweet in tweets:
#         # Tokenize
#         tokenized = word_tokenize(tweet.lower())
#         # Remove non-alphanumeric characters
#         tokenized_alpha = [t for t in tokenized if re.match("^[a-zA-Z]+$", t)]
#         # Remove stopwords after removing non-alphanumeric characters
#         tokenized_alpha_no_stopwords = [t for t in tokenized_alpha if t not in stopwords.words('english')]
#         # Re-form the tokens
#         new_review = " ".join(tokenized_alpha_no_stopwords)
#         # Append to new_tweets
#         new_tweets.append(new_review)
#     return np.array(new_tweets)
#
# print("Removing non-alpha and stopwords")
# data["training_tweets"] = no_alpha_no_stopwords(data["training_tweets"])
# data["testing_tweets"] = no_alpha_no_stopwords(data["testing_tweets"])

print("Vectorizing")
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(data["training_tweets"])
test_counts = count_vect.fit(data["testing_tweets"])

print("CLASSIFYING!")
clf_mlp = MLPClassifier()

print("FITTING!")
clf_mlp.fit(train_counts, data["training_tags"])

print("TESTING!")
score = clf_mlp.score(test_counts, data["testing_tags"])
print(f"MLP SCORE: {score}")