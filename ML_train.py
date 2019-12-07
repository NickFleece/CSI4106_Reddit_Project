import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pickle

###
### This file trains our model using the twitter data at the directories:
### "data/training_tweets.csv", "data/training_tags.csv", "data/testing_tweets.csv", "data/testing_tags.csv"
### It outputs the models and the vectorizer to the "/models" folder
###

data = {
    "training_tags": [],
    "testing_tags": [],
    "training_tweets": [],
    "testing_tweets": []
}
#load all of the files we need
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
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(data["training_tweets"])
test_counts = count_vect.transform(data["testing_tweets"])

print("CLASSIFYING!")
clf_mlp = MLPClassifier(verbose=True, max_iter=10)
clf_svm = SVC(verbose=True, gamma='auto')

print("Fitting MLP...")
clf_mlp.fit(train_counts, data["training_tags"])
print("Fitting SVM...")
clf_svm.fit(train_counts, data["training_tags"])

# Print some basic scores
print("Testing MLP...")
score = clf_mlp.score(test_counts, data["testing_tags"])
print(f"MLP SCORE: {score}")
print("Testing SVM...")
svm_score = clf_svm.score(test_counts, data["testing_tags"])
print(f"SVM SCORE: {svm_score}")

#save our models
with open("models/svm", 'wb') as file:
    pickle.dump(clf_svm, file)
with open("models/mlp", 'wb') as file:
    pickle.dump(clf_mlp, file)

#save our vectorizer for later
with open("models/vectorizer", 'wb') as file:
    pickle.dump(count_vect, file)