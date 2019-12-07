from pprint import pprint
import nltk
from reddit import getHotFromSubreddit
from preprocesscomments import preProcessCommentsFromPost
from praw.models.reddit.more import MoreComments
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer

###
### This is the file that runs our reddit data through the machine learning models as well as vader
###

#get our posts -- here's where we set the subreddit, number (maximum 1000), and keyword
posts = getHotFromSubreddit("politics", 100, "trump")

#This is the type of preprocessing on the reddit data
mode = "noEdit"

count = 0
comments = []
info = []
# loop through our posts
for post in posts:
    count += 1
    print(f"{count} / {len(posts)} : Parsing comments for post: {post.title}")
    # we can uncomment this but it takes a long time to run
    # post.comments.replace_more(limit=None)
    p_comments, p_info = preProcessCommentsFromPost(post.comments.list())
    #append the comment and it's info
    comments.append(p_comments)
    info.append(p_info)

###
### THIS IS VADER!
###

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('vader_lexicon')

sia = SIA()
vader_results = []
for p_comments, i in zip(comments, info):
    #loop through our comments in the desired preprocessed mode
    for comment, gilded, score in zip(p_comments[mode], i["gilded"], i["score"]):
        pol_score = sia.polarity_scores(comment)
        pol_score['comment_body'] = comment
        pol_score['score'] = score
        pol_score['gilded'] = gilded
        vader_results.append(pol_score)

vader_totals = {
    "count": 0,
    "pos": 0,
    "neg": 0
}
vader_counts = {
    "neu":0,
    "pos":0,
    "neg":0
}
for i in vader_results:
    max = 'neu'
    for j in ["pos", "neg"]:
        if i[j] > i[max]:
            max = j
    vader_counts[max] += 1
    if max != 'neu':
        vader_totals["pos"] += i['pos'] * i['score'] * (1 + (0.5 * i['gilded']))
        vader_totals["neg"] += i['neg'] * i['score'] * (1 + (0.5 * i['gilded']))
        vader_totals["count"] += i['score'] * (1 + (0.5 * i['gilded']))


###
### THIS IS SVM
###
with open("models/svm", 'rb') as svmFile:
    svm_clf = pickle.load(svmFile)

svm_results = []
count = 0
with open("models/vectorizer", 'rb') as file:
    count_vect = pickle.load(file)

for p_comments, i in zip(comments, info):
    comment_counts = count_vect.transform(p_comments[mode])
    svm_predictions = svm_clf.predict(comment_counts)
    # loop through our comments in the desired preprocessed mode
    for comment, gilded, score, tag in zip(p_comments[mode], i["gilded"], i["score"], svm_predictions):
        svm_results.append({
            "comment_body": comment,
            "gilded": gilded,
            "score": score,
            "tag": tag
        })

svm_counts = {
    "count": 0,
    "pos": 0,
    "neg": 0
}
svm_totals = {
    "pos": 0,
    "neg": 0
}
for result in svm_results:
    svm_counts["count"] += 1
    if result["tag"] == "4":
        svm_counts["pos"] += 1
        svm_totals["pos"] += result["score"] * (1 + result["gilded"])
    else:
        svm_counts["neg"] += 1
        svm_totals["neg"] += result["score"] * (1 + result["gilded"])

###
### THIS IS MLP
###
with open("models/mlp", 'rb') as file:
    mlp_clf = pickle.load(file)

mlp_results = []
for p_comments, i in zip(comments, info):
    comment_counts = count_vect.transform(p_comments[mode])
    mlp_predictions = mlp_clf.predict(comment_counts)
    # loop through our comments in the desired preprocessed mode
    for comment, gilded, score, tag in zip(p_comments[mode], i["gilded"], i["score"], mlp_predictions):
        mlp_results.append({
            "comment_body": comment,
            "gilded": gilded,
            "score": score,
            "tag": tag
        })

mlp_counts = {
    "count": 0,
    "pos": 0,
    "neg": 0
}
mlp_totals = {
    "pos": 0,
    "neg": 0
}
for result in mlp_results:
    mlp_counts["count"] += 1
    if result["tag"] == "4":
        mlp_counts["pos"] += 1
        mlp_totals["pos"] += result["score"] * (1 + result["gilded"])
    else:
        mlp_counts["neg"] += 1
        mlp_totals["neg"] += result["score"] * (1 + result["gilded"])

print(f"\n\nTOTAL COMMENTS PARSED: {svm_counts['count']}\n\n")

print("VADER:")
print(vader_counts)
print(vader_totals)

print("SVM:")
print(svm_counts)
print(svm_totals)

print("MLP:")
print(mlp_counts)
print(mlp_totals)

print(f"Total sentiment averaged (Vader): {(vader_totals['pos'] - vader_totals['neg']) / vader_totals['count']}")
print(f"Total sentiment averaged (SVM): {(svm_totals['pos'] - svm_totals['neg']) / (svm_totals['pos'] + svm_totals['neg'])}")
print(f"Total sentiment averaged (MLP): {(mlp_totals['pos'] - mlp_totals['neg']) / (mlp_totals['pos'] + mlp_totals['neg'])}")