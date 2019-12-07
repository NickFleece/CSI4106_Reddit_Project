import praw
from praw.models.reddit.more import MoreComments
import pandas as pd

###
### This file is the file that contacts the reddit api and loads a specific amount of comments from a subreddit where a title has a specific keyword
###

# This is the link to the reddit api
reddit = praw.Reddit(client_id='PhozOedzwljoYg',
                     client_secret='HcLvVLFkrEqDnqeAJIhYZ5ZHjb0',
                     user_agent='csi4106')

def getHotFromSubreddit(subreddit, qty, keyword=""):
    print(f"Getting {qty} posts from the subreddit r/{subreddit} with the keyword {keyword}")
    global reddit
    posts = []
    for result in reddit.subreddit(subreddit).hot(limit=qty):
        #check if our keyword is in the title
        if keyword.lower() in result.title.lower():
            posts.append(result)
    return posts