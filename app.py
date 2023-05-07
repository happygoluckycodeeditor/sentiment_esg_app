import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
from scipy.special import softmax
import matplotlib.pyplot as plt

st.markdown('''
# Sentiment Analysis Application
This is my text analysis and sentiment app
''')

#LoadingModel For analysis            
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

#Adding text files
st.write('Enter the text you want to analyse in the text box:')

text = st.text_area('Text for analysis:')

st.write('The text you have written is:', text )

if text:
    #Run on ROBERTA
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    scores_dict = {
        'Negative' : scores[0],
        'Neutral' : scores[1],
        'Positive' : scores[2]
    }

    Negative_sentiment = scores_dict['Negative']
    Neutral_sentiment = scores_dict['Neutral']
    Positive_sentiment = scores_dict['Positive']

    st.write('The probability scores are:')
    st.write('The Negative sentiment is:', scores_dict['Negative'])
    st.write('The Positive sentiment is:', scores_dict['Positive'])
    st.write('The Neutral sentiment is:', scores_dict['Neutral'])
else:
    st.write('No Text Composed')

st.header('Sentiment of your text on a graph:')
if text:
    #Run on ROBERTA
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    scores_dict = {
        'Negative' : scores[0],
        'Neutral' : scores[1],
        'Positive' : scores[2]
    }

    Negative_sentiment = scores_dict['Negative']
    Neutral_sentiment = scores_dict['Neutral']
    Positive_sentiment = scores_dict['Positive']

    sentiment_score = ['Negative Sentiment', 'Neutral Sentiment', 'Positive Sentiment']
    sentiment_p = [Negative_sentiment, Neutral_sentiment, Positive_sentiment]

    plt.bar(sentiment_score, sentiment_p)
    plt.title('Sentiment Towards SDG in Japan from a news Article')
    plt.xlabel('Type of Sentiment')
    plt.ylabel('Sentiment Probability')
    st.pyplot(plt.gcf())
    
else:
    st.write('No text Composed')
    


#Adding data files
st.header('You can also add Datasets (This section is not completed yet!)')
st.write('Upload data here:')

file_csv = st.file_uploader('Choose a CSV file to upload:')

if file_csv:
    df =  pd.read_csv(file_csv)
    st.write(df.head())
else:
    st.write('No file uploaded')