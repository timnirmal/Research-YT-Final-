# sentiment analysis
from textblob import TextBlob


# function to calculate subjectivity
def getSubjectivity(review):
    return TextBlob(review).sentiment.subjectivity


# function to calculate polarity
def getPolarity(review):
    return TextBlob(review).sentiment.polarity


# function to analyze the reviews
def analysis(score):
    # break into 5 categories
    if score < -0.6:
        return 'Strongly Negative'
    elif -0.6 <= score < -0.2:
        return 'Negative'
    elif -0.2 <= score < 0.2:
        return 'Neutral'
    elif 0.2 <= score < 0.6:
        return 'Positive'
    else:
        return 'Strongly Positive'


def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity, analysis(blob.sentiment.polarity)
