import praw
from praw.models.reddit.more import MoreComments
import pandas as pd

reddit = praw.Reddit(client_id='PhozOedzwljoYg',
                     client_secret='HcLvVLFkrEqDnqeAJIhYZ5ZHjb0',
                     user_agent='csi4106')

def getHotFromSubreddit(subreddit, qty):
    global reddit
    posts = []
    for result in reddit.subreddit(subreddit).hot(limit=qty):
        posts.append(result)
    return posts

hot_posts = getHotFromSubreddit("all", 10)

print("Counting comments...")
postCount = 1
data = []
for post in hot_posts:
    print(f"Parsing post {postCount} / {len(hot_posts)} : {post.title}")
    postCount += 1
    post.comments.replace_more(limit=None)
    print("Parsing comments...")
    commentCount = 0
    for comment in post.comments.list():
        if type(comment) == MoreComments:
            continue
        commentCount += 1
        if commentCount % 100 == 0:
            print(f"{commentCount} / {len(post.comments.list())} comments parsed", end="\r")
        data.append(
            {
                "body": comment.body,
                "score": comment.score,
                "total_awards_received": comment.total_awards_received
            }
        )

df  = pd.DataFrame(data)
print(df)
df.to_csv("test.csv")

# print(test)
# print(test2)
#
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
