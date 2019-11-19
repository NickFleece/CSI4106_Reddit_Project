#pip install praw

import praw



reddit = praw.Reddit(client_id='PhozOedzwljoYg',
                     client_secret='HcLvVLFkrEqDnqeAJIhYZ5ZHjb0',
                     user_agent='csi4106')


def getHotFromSubreddit(subreddit, qty):
    global reddit
    posts = []
    comments = []
    for result in reddit.subreddit(subreddit).hot(limit=qty):
        posts.append(result)
        comments.append(result.comments.list())
    return posts, comments
test, test2 = getHotFromSubreddit("rocketleagueesports",10)

print(test)
print(test2)

for i in test:
    #print(i.__dict__.keys()) #gives you a list of all the attributes
    print(i.title)
    print(i.gilded)
    print(i.score)
    print(i.gildings)
for i in test2:
    print(i.__dict__.keys())
    for j in i:
        print(j.body)
        print(j.score)
        print(j.total_awards_received)
        print(j.gilded)
        print(j.edited)
        #print(j.html)
