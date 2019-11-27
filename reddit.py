import praw
from praw.models.reddit.more import MoreComments
import pandas as pd

reddit = praw.Reddit(client_id='PhozOedzwljoYg',
                     client_secret='HcLvVLFkrEqDnqeAJIhYZ5ZHjb0',
                     user_agent='csi4106')

def getHotFromSubreddit(subreddit, qty, keyword=""):
    print(f"Getting {qty} posts from the subreddit r/{subreddit} with the keyword {keyword}")
    global reddit
    posts = []
    for result in reddit.subreddit(subreddit).hot(limit=qty):
        if keyword.lower() in result.title.lower():
            posts.append(result)
    return posts

def exportCommentsToCSV(posts):
    print("Counting comments...")
    postCount = 1
    data = []
    for post in posts:
        print(f"Parsing post {postCount} / {len(posts)} : {post.title}")
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

# hot_posts = getHotFromSubreddit("all", 10)