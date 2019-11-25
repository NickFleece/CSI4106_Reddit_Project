from pprint import pprint
import nltk
from reddit import getHotFromSubreddit
from praw.models.reddit.more import MoreComments
import matplotlib.pyplot as plt

posts = getHotFromSubreddit("news", 1)

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('vader_lexicon')

sia = SIA()
results = []

# for post in posts:
#     pol_score = sia.polarity_scores(post.title)
#     pol_score['headline'] = post.title
#     results.append(pol_score)

count = 0
for post in posts:
    count += 1
    print(f"{count} / {len(posts)} : Parsing comments for post {post.title}")
    for comment in post.comments.list():
        if type(comment) == MoreComments:
            continue
        else:
            pol_score = sia.polarity_scores(comment.body)
            pol_score['comment_body'] = comment.body
            results.append(pol_score)

# pprint(results, width=100)

# for i in test:
#     #print(i.__dict__.keys()) #gives you a list of all the attributes
#     print(i.title)
#     print(i.gilded)
#     print(i.score)
#     print(i.gildings)
# for i in test2:
#     print(i.__dict__.keys())
#     for j in i:
#         print(j.body)
#         print(j.score)
#         print(j.total_awards_received)
#         print(j.gilded)
#         print(j.edited)
#         #print(j.html)
