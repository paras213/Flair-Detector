from django.shortcuts import render
from django.http import HttpResponse
import datetime
import csv
import praw
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
sub_name = ['india']
post_num = 250
reddit = praw.Reddit(
                     client_id='hDIJdjc2k5VzTA',
                     client_secret='fT9JnoUGAUAqDipudbAZ1XgeLFM',
                     username='HelloWorld13211',
                     password='Qwerty@123',
                     user_agent='web scrapper'
                     )
def write_csv(to_csv, csv_name):
    try:
        with open(csv_name, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(to_csv)
            file.close()
    except Exception as e:
        print(e)
def retrieve_post(sub,post_number,csv_name):
    subreddit = reddit.subreddit(sub)
    sub_hot = subreddit.hot(limit=post_number)
    to_csv = [['flair_name','title']]

    print('Scrapping {}...'.format(sub))
    try:
        for post in sub_hot:
            if not post.stickied:
                retrieved = [
                            post.link_flair_text,
                            post.title,
                            ]
            
                to_csv.append(retrieved)
           
        write_csv(to_csv, csv_name)
    except Exception as e:
        print(e)
for sub in sub_name:
    csv_name = '{}.csv'.format(sub)
    csv_file = Path(csv_name)
    if csv_file.exists():
        print('File {} already exist!'.format(csv_name))
    else:
        retrieve_post(sub,post_num,csv_name)
    print("done")
df=pd.read_csv("india.csv ")
df.flair_name=pd.Categorical(df.flair_name)
df['flair']=df.flair_name.cat.codes 
data_x=df.title
data_y=df.flair
cv=TfidfVectorizer(min_df=1)
X=cv.fit_transform(data_x)
Y=data_y.astype(int)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=4)
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
def home(request):
    if request.method=="POST":
        data=request.POST['kuchbhi']
        x=predict(data)
        return render(request,"flair/index.html",{'data':x})
    return render(request,"flair/index.html")

def predict(url):
    submission = reddit.submission(url=url)
    title=cv.transform([submission.title])
    index=int(mnb.predict(title))
    result=df.flair_name.cat.categories[index]
    return result
