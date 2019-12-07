import pandas as pd
import csv
import random
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
import re

###
### This is the file that preprocesses the tweets, it splits the training and testing data as well as removing any '@' characters
###

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

    # shuffle the results
    random.shuffle(results)

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
        if index % 10000 == 0:
            print(f"{index} rows parsed")
        # if index > training_testing_ratio * len(results):
        tweet = preprocessTweet(row[5])
        # we only want 100,000 tweets
        if index > 100000:
            split_data["testing_tags"].append(row[0])
            split_data["testing_tweets"].append(row[5])
        else:
            #this will only get 20,000 for our testing set
            split_data["training_tags"].append(row[0])
            split_data["training_tweets"].append(row[5])

        #break after 120,000
        if index > 120000:
            break

    for key in split_data.keys():
        print(f"Exporting {key}")
        df = pd.DataFrame(split_data[key])
        #export all of the data to their own csv files
        df.to_csv(f"data/{key}.csv")

def preprocessTweet(tweet):
    result = word_tokenize(tweet)
    #remove @ and the corresponding user
    for i in range(0, len(result)):
        if result[i] == '@' and i != len(result) - 1:
            result.pop(i + 1)
            result.pop(i)
            break

    return " ".join(result)

process_raw()