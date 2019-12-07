# CSI4106_Reddit_Project

Run our files in this order in order to correctly process and train our models

- tweet_preprocessing.py: Preprocess our twitter data, we need [this](https://www.kaggle.com/kazanova/sentiment140) dataset at the directory data/twitter.csv, it splits our data into testing and training pairs
- ML_train.py: This trains our two models (SVM, MLP) and prints out some initial scores
- ML_test.py: This file tests on the testing set and prints detailed results (recall, precision, accuracy)
- headline.py: This file is the one that runs our model on the reddit data that we get from the api, currently by default it is set up to run on the subreddit "/r/politics" with the keyword "Trump"

These are helper files:

- reddit.py: This is the file that connects our python client to the reddit api and collects the posts with the keyword in the title
- preprocesscomments.py: This file preprocesses the comments, all of the methods were outlined in the report
