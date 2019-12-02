from praw.models import MoreComments
from reddit import getHotFromSubreddit
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import word_tokenize

print(str("a"))
test, test2 = getHotFromSubreddit("news",1)


def preProcessCommentsFromPost(post):
    stemmer = PorterStemmer()
    preprocessedCommentsBody = {}
    preprocessedCommentsBody["noEdit"] = []
    preprocessedCommentsBody["stemmed"] = []
    preprocessedCommentsBody["noStopWords"] = []
    preprocessedCommentsBody["alphaNumOnly"] = []
    preprocessedCommentsBody["VerbsOnly"] = []
    preprocessedCommentsBody["NoVerbs"] = []
    preprocessedCommentsInfo = {}
    preprocessedCommentsInfo["score"] = []
    preprocessedCommentsInfo["gilded"] = []
    preprocessedCommentsInfo["total_awards_received"] = []
    for comment in post:
        if (str(comment.author) != "AutoModerator" and "bot" not in str(comment.author)): #we don't take comments made by bots or AutoModerator as they're usually useless.
            preprocessedCommentsInfo["score"].append(comment.score)
            preprocessedCommentsInfo["gilded"].append(comment.gilded)
            preprocessedCommentsInfo["total_awards_received"].append(comment.total_awards_received)
            preprocessedCommentsBody["noEdit"].append(comment.body)


            #make the tokens very similar to notebook 5 making the assumption that all comments are english
            #replace links with nothing as links are neutral.
            commentsNoLinks = comment.body
            commentslinks = re.findall("http[^\s]*",comment.body)

            for link in commentslinks:
                commentsNoLinks = commentsNoLinks.replace(link,"")

            #remove reddit formatting
            commentsNoLinks = commentsNoLinks.replace("\n"," ")
            commentsNoLinks = commentsNoLinks.replace("..."," ")
            commentsNoLinks = commentsNoLinks.replace("x200b"," ")
            test = commentsNoLinks.maketrans("()<>[]*?&#`''\\", " "*len("()<>[]*?&#`''\\"))

            commentsNoLinks = commentsNoLinks.translate(test)

            tokens = word_tokenize(commentsNoLinks.lower())
            #making sure that for the machine learnign algs that every word is a similar as possible (all caps might indicate more emotion tho but adds variability between inputs)
            tokens_stemmed = [stemmer.stem(t) for t in  tokens]
            tokens_nostopwords = [t for t in tokens_stemmed if t not in stopwords.words("english")]
            tokens_pos = nltk.pos_tag(tokens_nostopwords)

            preprocessedCommentsBody["stemmed"].append(" ".join(tokens_stemmed))
            preprocessedCommentsBody["noStopWords"].append(" ".join(tokens_nostopwords))
            preprocessedCommentsBody["alphaNumOnly"].append(" ".join([x for x in tokens_nostopwords if re.match("^[a-zA-Z]+$",x)]))
            preprocessedCommentsBody["VerbsOnly"].append(" ".join([p[0] for p in tokens_pos if p[1].startswith("V")]))
            preprocessedCommentsBody["NoVerbs"].append(" ".join([p[0] for p in tokens_pos if not p[1].startswith("V")]))

    return preprocessedCommentsBody, preprocessedCommentsInfo;

for i in test2:
    print(i)
    c1,c2 = preProcessCommentsFromPost(i)
    print(c1)
