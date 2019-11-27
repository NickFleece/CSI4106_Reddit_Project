from pprint import pprint
import nltk
from reddit import getHotFromSubreddit
from praw.models.reddit.more import MoreComments
import matplotlib.pyplot as plt

posts = getHotFromSubreddit("politics", 100, "Trump")

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('vader_lexicon')

sia = SIA()
results = []

count = 0
for post in posts:
    count += 1
    print(f"{count} / {len(posts)} : Parsing comments for post: {post.title}")
    # post.comments.replace_more(limit=None)
    for comment in post.comments.list():
        if type(comment) == MoreComments:
            continue
        else:
            pol_score = sia.polarity_scores(comment.body)
            pol_score['comment_body'] = comment.body
            pol_score['score'] = comment.score
            pol_score['gilded'] = comment.gilded
            results.append(pol_score)

# pprint(results[:50], width=1000)

# x = []
# y = []
# for i in results:
#     x.append(i['pos'])
#     y.append(i['neg'])
# plt.scatter(x,y)
# plt.xlabel("pos")
# plt.ylabel("neg")
# plt.show()

totals = {
    "count": 0,
    "pos": 0,
    "neg": 0
}
counts = {
    "neu":0,
    "pos":0,
    "neg":0
}
for i in results:
    max = 'neu'
    for j in ["pos", "neg"]:
        if i[j] > i[max]:
            max = j
    counts[max] += 1
    if max != 'neu':
        totals["pos"] += i['pos'] * i['score'] * (1 + (0.5 * i['gilded']))
        totals["neg"] += i['neg'] * i['score'] * (1 + (0.5 * i['gilded']))
        totals["count"] += i['score'] * (1 + (0.5 * i['gilded']))

print(counts)
print(totals)
print(f"Total sentiment averaged: {(totals['pos'] - totals['neg']) / totals['count']}")