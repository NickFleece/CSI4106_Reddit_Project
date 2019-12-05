import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer

data = {
    "testing_tags": [],
    "testing_tweets": [],
    "training_tweets": []
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

print("Vectorizing")
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(data["training_tweets"])
test_counts = count_vect.transform(data["testing_tweets"])

models = {
    "svm": None,
    "mlp": None
}
for i in models.keys():
    with open(f"models/{i}", 'rb') as pickle_file:
        models[i] = pickle.load(pickle_file)

predictions = {}

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