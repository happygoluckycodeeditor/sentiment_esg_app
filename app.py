import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
from scipy.special import softmax
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.markdown('''
# Sentiment Analysis Application
This is my text analysis and sentiment app
''')

#LoadingModel For analysis            
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Added Caching
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    return tokenizer, model

tokenizer, model = load_model()

#Adding text files
st.write('Enter the text you want to analyse in the text box:')

@st.cache_data(max_entries=3)
def cache_input_text(text):
    return text

text = st.text_area('Text for analysis:')
submit_button = st.button('Submit')

st.write('The text you have written is:')
st.write(text)

if submit_button or text:
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
        plt.title('Sentiment of the text')
        plt.xlabel('Type of Sentiment')
        plt.ylabel('Sentiment Probability')
        st.pyplot(plt.gcf())
        
    else:
        st.write('No text Composed')

    st.header('The Worcloud of your Text is as follows:')
    if text:
        stop_w = set(STOPWORDS)
        wordcloud = WordCloud(stopwords = stop_w, width=800, height=800).generate(text)

        #Displaying the image
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Wordcloud of the text')
        plt.axis("off")
        st.pyplot(plt.gcf())
    else:
        st.write('No Text Composed')
    


#Adding data files
st.header('You can also add Datasets (This section is not completed yet!)')
st.write('Upload data here:')

file_csv = st.file_uploader('Choose a CSV file to upload:')

if file_csv:
    df =  pd.read_csv(file_csv)
    st.write(df.head())
else:
    st.write('No file uploaded')


st.subheader('App by Tanmay Bagwe')
st.subheader(body='For any questions send an email at tanmay.bagwe.tb@gmail.com')