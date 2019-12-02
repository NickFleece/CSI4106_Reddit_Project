import pandas as pd
import csv
import random

def process_raw():
    results = []
    with open("data/twitter.csv") as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=",")
        lineCount = 0
        for row in csv_reader:
            results.append(row)
            lineCount += 1
            if lineCount % 100000 == 0:
                print(f"{lineCount} lines parsed")

    random.shuffle(results)
    # data = pd.DataFrame(results)

    split_data = {
        "training_tags":[],
        "testing_tags":[],
        "training_tweets":[],
        "testing_tweets":[]
    }
    training_testing_ratio = 0.8
    index = 0
    for row in results:
        index += 1
        if index % 100000 == 0:
            print(f"{index} rows parsed")
        # if index > training_testing_ratio * len(results):
        if index > 10000:
            split_data["testing_tags"].append(row[0])
            split_data["testing_tweets"].append(row[5])
        else:
            split_data["training_tags"].append(row[0])
            split_data["training_tweets"].append(row[5])

        if index > 12000:
            break

    for key in split_data.keys():
        print(f"Exporting {key}")
        df = pd.DataFrame(split_data[key])
        df.to_csv(f"data/{key}.csv")

process_raw()