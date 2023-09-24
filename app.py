import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
from scipy.special import softmax
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

MAX_LENGTH = 512 #For Truncation of the BERT Models

st.set_page_config(initial_sidebar_state="auto")
st.markdown('''
# Sentiment Analysis Application and SDG Classifier
This is my text analysis and sentiment app
''')

#SideBar_Config
st.sidebar.title('Sentiment Analysis and SDG Classifier')
st.sidebar.caption('This app is made as a proof of concept for my upcoming master thesis. The app uses Modifiied BERT Models from Hugging face namely, roBERTa Sentiment and OSDG model for sentiment classification and SDG Text Classification')
st.sidebar.caption('The App can detect text in upto 16 Sustainable Development Goals and is also able to tell what the sentiment of the text input.')

image = Image.open('Favicon.png')
st.sidebar.image(image)
st.sidebar.subheader('For any feedback or question please contact me on my email at tanmay.bagwe.tb@gmail.com')
st.sidebar.caption('Last updated 15th May,2023')

#LoadingModel For analysis            
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Added Caching
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    return tokenizer, model

tokenizer, model = load_model()

@st.cache_resource
def load_model2():
    tokenizer2 = AutoTokenizer.from_pretrained("jonas/bert-base-uncased-finetuned-sdg-Mar23")
    model2 = AutoModelForSequenceClassification.from_pretrained("jonas/bert-base-uncased-finetuned-sdg-Mar23")
    return tokenizer2, model2

tokenizer2, model2 = load_model2()

#Adding text files
st.write('Enter the text you want to analyse in the text box:')
st.write('このアプリは入力した文章のセンチメント分析（感情）とSDG分類を行うことができます')

@st.cache_data(ttl=500)
def cache_input_text(text):
    return text

text = st.text_area('Text for analysis:')
submit_button = st.button('Submit')

st.write('The text you have written is:')
st.write(text)

if submit_button or text:
    if text:
        #Run on ROBERTA
        encoded_text = tokenizer(text, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
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
        encoded_text = tokenizer(text, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
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
        plt.xticks(rotation=60)
        st.pyplot(plt.gcf(), clear_figure=True)
        
    else:
        st.write('No text Composed')

    st.header('Which SDG is the Text talking about?')
    if text:
        #Run on SDG
        encoded_text = tokenizer2(text, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
        output = model2(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        scores_dict2 = {
            'SDG1' : scores[0],
            'SDG10' : scores[1],
            'SDG11' : scores[2],
            'SDG12' : scores[3],
            'SDG13' : scores[4],
            'SDG14' : scores[5],
            'SDG15' : scores[6],
            'SDG16' : scores[7],
            'SDG2' : scores[8],
            'SDG3' : scores[9],
            'SDG4' : scores[10],
            'SDG5' : scores[11],
            'SDG6' : scores[12],
            'SDG7' : scores[13],
            'SDG8' : scores[14],
            'SDG9' : scores[15]
        }

        SDG1 = scores_dict2['SDG1'] * 100
        SDG2 = scores_dict2['SDG2'] * 100
        SDG3 = scores_dict2['SDG3'] * 100
        SDG4 = scores_dict2['SDG4'] * 100
        SDG5 = scores_dict2['SDG5'] * 100
        SDG6 = scores_dict2['SDG6'] * 100
        SDG7 = scores_dict2['SDG7'] * 100
        SDG8 = scores_dict2['SDG8'] * 100
        SDG9 = scores_dict2['SDG9'] * 100
        SDG10 = scores_dict2['SDG10'] * 100
        SDG11 = scores_dict2['SDG11'] * 100
        SDG12 = scores_dict2['SDG12'] * 100
        SDG13 = scores_dict2['SDG13'] * 100
        SDG14 = scores_dict2['SDG14'] * 100
        SDG15 = scores_dict2['SDG15'] * 100
        SDG16 = scores_dict2['SDG16'] * 100

        sentiment_score_b = ['SDG1', 'SDG2', 'SDG3', 'SDG4', 'SDG5', 'SDG6', 'SDG7', 'SDG8', 'SDG9', 'SDG10', 'SDG11', 'SDG12', 'SDG13', 'SDG14', 'SDG15', 'SDG16']
        sentiment_p_b = [SDG1, SDG2, SDG3, SDG4, SDG5, SDG6, SDG7, SDG8, SDG9, SDG10, SDG11, SDG12, SDG13, SDG14, SDG15, SDG16]
        
        plt.bar(sentiment_score_b, sentiment_p_b)
        plt.title('SDG Scores')
        plt.xlabel('Type of SDG')
        plt.ylabel('SDG Probability')
        plt.xticks(rotation=60)
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