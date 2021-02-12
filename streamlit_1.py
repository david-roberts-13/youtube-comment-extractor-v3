import streamlit as st
import pandas as pd
import numpy as np

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import urllib.parse as p
import re
import os
import pickle
import spacy

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import unicodedata
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import urllib.parse as p
import os

#Text Preprocessing 
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


client= {"installed":{"client_id":"552781266117-lmvt9f6566pa9h6f4h75v29e19h311va.apps.googleusercontent.com","project_id":"youtube-comment-project-v3","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"jSz-cLr3CC0av9NUAXKpptAt","redirect_uris":["urn:ietf:wg:oauth:2.0:oob","http://localhost"]}}
#test12


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

with header:
	st.title('Welcome to my awesome data scient project')
	st.header("Youtube Scraper")
	st.text('in this project i wanted to make a tool that would aggregate youtube comment sentiment')
	url_input = st.text_input('Enter Youtube Video Link')
	client_secret=client
	SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

	Video_URL_Input=url_input
#-----------------------------------------------------------------------------------------------------------------------

def youtube_authenticate():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = client_secret
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials availablle, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            #flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            flow = InstalledAppFlow.from_client_config(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build(api_service_name, api_version, credentials=creds)

# authenticate to YouTube API
youtube = youtube_authenticate()

#-----------------------------------------------------------------------------------------------------------------------
def get_channel_id_by_url(youtube, url):
    """
    Returns channel ID of a given `id` and `method`
    - `method` (str): can be 'c', 'channel', 'user'
    - `id` (str): if method is 'c', then `id` is display name
        if method is 'channel', then it's channel id
        if method is 'user', then it's username
    """
    # parse the channel URL
    method, id = parse_channel_url(url)
    if method == "channel":
        # if it's a channel ID, then just return it
        return id
    elif method == "user":
        # if it's a user ID, make a request to get the channel ID
        response = get_channel_details(youtube, forUsername=id)
        items = response.get("items")
        if items:
            channel_id = items[0].get("id")
            return channel_id
    elif method == "c":
        # if it's a channel name, search for the channel using the name
        # may be inaccurate
        response = search(youtube, q=id, maxResults=1)
        items = response.get("items")
        if items:
            channel_id = items[0]["snippet"]["channelId"]
            return channel_id
    raise Exception(f"Cannot find ID:{id} with {method} method")

#-----------------------------------------------------------------------------------------------------------------------


def get_video_id_by_url(url):
    """
    Return the Video ID from the video `url`
    """
    # split URL parts
    parsed_url = p.urlparse(url)
    # get the video ID by parsing the query of the URL
    video_id = p.parse_qs(parsed_url.query).get("v")
    if video_id:
        return video_id[0]
    else:
        raise Exception(f"Wasn't able to parse video URL: {url}")
#-----------------------------------------------------------------------------------------------------------------------

def get_video_details(youtube, **kwargs):
    return youtube.videos().list(
        part="snippet,contentDetails,statistics",
        **kwargs
    ).execute()
#-----------------------------------------------------------------------------------------------------------------------

df_title = pd.DataFrame(columns=['title'])
def print_video_infos(video_response):
    items = video_response.get("items")[0]
    # get the snippet, statistics & content details from the video response
    snippet         = items["snippet"]
    statistics      = items["statistics"]
    content_details = items["contentDetails"]
    # get infos from the snippet
    channel_title = snippet["channelTitle"]
    title         = snippet["title"]
    description   = snippet["description"]
    publish_time  = snippet["publishedAt"]
    # get stats infos
    comment_count = statistics["commentCount"]
    like_count    = statistics["likeCount"]
    dislike_count = statistics["dislikeCount"]
    view_count    = statistics["viewCount"]
    # get duration from content details
    duration = content_details["duration"]
    # duration in the form of something like 'PT5H50M15S'
    # parsing it to be something like '5:50:15'
    parsed_duration = re.search(f"PT(\d+H)?(\d+M)?(\d+S)", duration).groups()
    duration_str = ""
    for d in parsed_duration:
        if d:
            duration_str += f"{d[:-1]}:"
    duration_str = duration_str.strip(":")
    df_title = pd.DataFrame(columns=['title'])
    df_channel = pd.DataFrame(columns=['channel_title'])
    df_views = pd.DataFrame(columns=['view_count'])
    df_likes = pd.DataFrame(columns=['like_count'])
    df_dislikes = pd.DataFrame(columns=['dislike_count'])

    
    
    df_title = df_title.append({'title': title},ignore_index=True)
    df_channel = df_channel.append({'channel_title': channel_title},ignore_index=True)
    df_views = df_views.append({'view_count': view_count},ignore_index=True)
    df_likes = df_likes.append({'like_count': like_count},ignore_index=True)
    df_dislikes = df_dislikes.append({'dislike_count': dislike_count},ignore_index=True)
    
    frames = [df_title,df_channel,df_views,df_likes,df_dislikes]
    df_meta=pd.concat(frames,axis=1)
    return(df_meta)
#-----------------------------------------------------------------------------------------------------------------------

def get_comments(youtube, **kwargs):
    return youtube.commentThreads().list(
        part="snippet",
        **kwargs
    ).execute()
#-----------------------------------------------------------------------------------------------------------------------
url = Video_URL_Input
df_comments = pd.DataFrame(columns=['textOriginal'])
#df_title1 = pd.DataFrame(columns=['title1'])
df_update = pd.DataFrame(columns=['Update'])
df_like = pd.DataFrame(columns=['Like_Count'])
df_comid = pd.DataFrame(columns=['Comment_ID'])
df_comments_original=pd.DataFrame(columns=['Comment_original'])
if "watch" in url:
    # that's a video
    video_id = get_video_id_by_url(url)
    params = {
        'videoId': video_id, 
        'maxResults': 100,
        'order': 'relevance', # default is 'time' (newest)
    }
else:
    # should be a channel
    channel_id = get_channel_id_by_url(url)
    params = {
        'allThreadsRelatedToChannelId': channel_id, 
        'maxResults': 100,
        'order': 'relevance', # default is 'time' (newest)
    }
# get the first 2 pages (2 API requests)
n_pages = 50
for i in range(n_pages):
    # make API call to get all comments from the channel (including posts & videos)
    response = get_comments(youtube, **params)
    items = response.get("items")
    # if items is empty, breakout of the loop
    if not items:
        break
    for item in items:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        #title1 = item["snippet"]["topLevelComment"]["snippet"]["title"]
        updated_at = item["snippet"]["topLevelComment"]["snippet"]["updatedAt"]
        like_count = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
        comment_id = item["snippet"]["topLevelComment"]["id"]
        
        df_comments = df_comments.append({'textOriginal': comment},ignore_index=True)
        #df_title1 = df_title1.append({'title1': title1},ignore_index=True)
        df_update = df_update.append({'Update': updated_at},ignore_index=True)
        df_like = df_like.append({'Like_Count': like_count},ignore_index=True)
        df_comid = df_comid.append({'Comment_ID': comment_id},ignore_index=True)
        df_comments_original = df_comments_original.append({'Comment_original': comment},ignore_index=True)

    if "nextPageToken" in response:
        # if there is a next page
        # add next page token to the params we pass to the function
        params["pageToken"] =  response["nextPageToken"]
    else:
        # must be end of comments!!!!
        break
    print("*"*70)



frames = [df_comments, df_update, df_like,df_comid,df_comments_original]
df=pd.concat(frames,axis=1)



#-----------------------------------------------------------------------------------------------------------------------
def smiley(a):
    x1=a.replace(":‑)","happy")
    x2=x1.replace(";)","happy")
    x3=x2.replace(":-}","happy")
    x4=x3.replace(":)","happy")
    x5=x4.replace(":}","happy")
    x6=x5.replace("=]","happy")
    x7=x6.replace("=)","happy")
    x8=x7.replace(":D","happy")
    x9=x8.replace("xD","happy")
    x10=x9.replace("XD","happy")
    x11=x10.replace(":‑(","sad")    #using 'replace' to convert emoticons
    x12=x11.replace(":‑[","sad")
    x13=x12.replace(":(","sad")
    x14=x13.replace("=(","sad")
    x15=x14.replace("=/","sad")
    x16=x15.replace(":[","sad")
    x17=x16.replace(":{","sad")
  
    x18=x17.replace(":P","playful")
    x19=x18.replace("XP","playful")
    x20=x19.replace("xp","playful")
  
    
    x21=x20.replace("<3","love")
    x22=x21.replace(":o","shock")
    x23=x22.replace(":-/","sad")
    x24=x23.replace(":/","sad")
    x25=x24.replace(":|","sad")
    return x25
df['emoticons_replacment']=df['textOriginal'].apply(smiley)
#-----------------------------------------------------------------------------------------------------------------------


df["less_spaces"]=df['emoticons_replacment'].apply(lambda x: re.sub(' +', ' ', x))


#https://towardsdatascience.com/preprocessing-text-data-using-python-576206753c28 

df['text_expan_contractions'] = df['less_spaces'].apply(lambda x: [contractions.fix(word) for word in x.split()])
df['text_expan_contractions'] = [' '.join(map(str, l)) for l in df['text_expan_contractions']]

#removes non alphanumeric/ whitespace characters from strings 
df['text_misc_char_removed'] = df['text_expan_contractions'].str.replace('&#39;','')  # just a lil something to replace the weird apostroph thing 
df['text_misc_char_removed'] = df['text_misc_char_removed'].map(lambda x: re.sub("[^0-9a-zA-Z\s]+",'', x)) #this includes puncutation which shoes little value in analysis 

#removes emojis 

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

df['text_no-emoji']= df['text_misc_char_removed'].apply(lambda x:deEmojify(x)) 

#lower case
df['lower']=df['text_no-emoji'].str.lower() # lower must come before contractions since its matching based

#tokenizing text into new column 
df['text_tokenized'] = df['lower'].apply(word_tokenize)

#removing stop words like 'you, he, she, in, a, has'
stop_words = set(stopwords.words('english'))
df['stop_words_removed'] = df['text_tokenized'].apply(lambda x: [word for word in x if word not in stop_words])

#The idea of stemming is to reduce different forms of word usage into its root word. For example, “drive”, 
#“drove”, “driving”, “driven”, “driver” are derivatives of the word “drive” and very often researchers want 
#to remove this variability from their corpus. Compared to lemmatization, stemming is certainly the less 
#complicated method but it often does not produce a dictionary-specific morphological root of the word. 
#In other words, stemming the word “pies” will often produce a root of “pi” whereas lemmatization will 
#find the morphological root of “pie”.

#Lemmatization
df['pos_tags'] = df['stop_words_removed'].apply(nltk.tag.pos_tag)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])


wnl = WordNetLemmatizer()
df['lemmatized'] = df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])



#converting back into string for temporary analysis 
def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(s)) 

df['final_text']= df['lemmatized'].apply(lambda x:listToString(x)) 

#-----------------------------------------------------------------------------------------------------------------------


with dataset:
	from textblob import TextBlob 

def get_comment_sentiment(comment): 
	''' 
	Function to return sentiment score of each comment
	'''
	analysis = TextBlob(comment) 
	return analysis.sentiment.polarity

sentiment = df['final_text'].apply(get_comment_sentiment)
df['Sentiment'] = sentiment




with features:

	st.header('Proccessed Comments')
	st.text('As you from left to right in this data frame you will see each step taken in preprocessing')
	st.dataframe(df.tail(7))


import plotly.graph_objects as go
import numpy as np
import plotly.express as px


fig = go.Figure()
# Use x instead of y argument for horizontal plot


fig.add_trace(go.Box(x=df['Sentiment']))
fig = px.box(df, x="Sentiment", points="all",hover_data=['Comment_original'])


fig.update_layout(
    height=300,
    title_text='Youtube Comment Sentiment analysis',showlegend=False)

fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='Sentiment_bin')),
                  selector=dict(mode='markers'))

st.header('Initial Visualization')

st.plotly_chart(fig)
