# %% [markdown]
# # Natural Lamguage Processing

# %% [markdown]
# 

# %% [markdown]
# # Spam/Ham classification using NLP

# %%
import nltk
import pandas as pd
import numpy as np

# %%
import pandas as pd

data = {
    "label": ["spam"] * 50 + ["ham"] * 50,
    "text": [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "URGENT! Your account has been compromised. Reset your password immediately!",
        "Get rich quick! Earn $10,000 per week working from home.",
        "Exclusive offer for you: Buy 1 get 1 free. Limited time only!",
        "You have an unclaimed reward waiting for you. Act fast!",
        "Win a free vacation to the Bahamas! Just enter your details.",
        "Final notice: Your loan approval is pending. Apply now!",
        "Your Netflix account is suspended. Click this link to reactivate.",
        "You've been selected for a $500 Walmart gift card!",
        "Hot deal: 70% off all electronics! Shop now.",
        "Claim your free cryptocurrency airdrop before it's too late!",
        "Important: Your social security number has been flagged. Contact us now!",
        "Earn $5,000 weekly by working just 2 hours a day!",
        "Limited offer: Subscribe now and get a free 1-year membership!",
        "Congratulations, you've won a free cruise! Confirm your details.",
        "Your PayPal account has been locked. Click here to secure it.",
        "Special promotion: Save 50% on all purchases this weekend!",
        "Your Amazon order needs verification. Click here to confirm.",
        "Exclusive deal: Get a $1000 gift card by completing this survey.",
        "Investment opportunity: Double your money in just 24 hours!",
        "Act now! This limited-time mortgage offer expires soon.",
        "Hurry! Your free trial is about to expire. Renew today!",
        "Click here to unlock secret discounts on luxury brands.",
        "Final reminder: Your extended car warranty is about to expire!",
        "You are a lucky winner of a brand-new Tesla Model S!",
        "Massive discount on Ray-Ban sunglasses! Buy now!",
        "Dear user, your ATM card has been blocked. Click to reactivate.",
        "Make $500 daily with zero investment! Click to learn more.",
        "Your bank account is under review. Update details now!",
        "Congratulations! You've won a new MacBook Pro. Claim now!",
        "Instant approval: Get a credit card with no credit check!",
        "Breaking news: This stock will skyrocket soon! Invest today!",
        "Download this app and earn money while you sleep!",
        "Your phone number has been selected for a $1,000 reward!",
        "Limited-time deal: 90% off on all premium software!",
        "Earn unlimited cash by signing up for our referral program!",
        "Alert! Your system has been infected with a virus. Click to fix.",
        "Last chance to claim your holiday prize! Don't miss out.",
        "Special giveaway: Win an all-expenses-paid trip to Dubai!",
        "Click now to access your free government grant money!",
        "You've been pre-approved for a $50,000 personal loan!",
        "Hurry! Your credit score is dropping. Fix it now!",
        "Free Bitcoin giveaway! Get yours before it's gone!",
        "Act fast! Your unpaid toll charges need immediate payment.",
        "Dear user, confirm your email to receive a $500 bonus!",
        "Be a millionaire overnight! Sign up for this program today!",
        "Increase your followers instantly with this new tool!",
        "Shocking secret revealed: Doctors hate this simple trick!",
        "Get your free trial of this amazing weight loss supplement!",
        "Hi, how are you doing today?",
        "Are we still on for lunch tomorrow?",
        "Mom, I'll call you back in 10 minutes.",
        "Can you send me the report by EOD?",
        "Hey, did you watch the game last night?",
        "Meeting at 3 PM in the conference room.",
        "Your package has been delivered. Check your mailbox.",
        "Happy birthday! Hope you have a great day!",
        "Let's catch up over coffee this weekend.",
        "How was your trip to Italy?",
        "Just checking in. Haven't heard from you in a while.",
        "I'll pick you up from the airport at 6 PM.",
        "Don't forget to submit your assignment by Friday.",
        "Hope you are feeling better today.",
        "Dinner at my place tonight. Let me know if you can make it.",
        "See you at the gym later!",
        "Your order has been shipped. Tracking details enclosed.",
        "Great job on the presentation today!",
        "Let's plan a weekend getaway soon.",
        "Need help with anything? Just let me know.",
        "Remember to pay the electricity bill before due date.",
        "Hey, I found this book you might like!",
        "Looking forward to our hiking trip next month.",
        "Check your email for the latest updates.",
        "Let’s schedule a call to discuss the project.",
        "Happy anniversary! Hope you have a wonderful day.",
        "Thanks for your help with the project!",
        "Got your message. I'll call you back soon.",
        "Don't forget about our dinner reservation at 7 PM.",
        "Excited for the concert this weekend!",
        "I'll be late to the meeting, traffic is bad.",
        "Let's go for a run in the morning.",
        "See you at the team event tomorrow.",
        "Hope you had a fantastic weekend!",
        "Have you seen the latest movie release?",
        "Check out this new cafe I found!",
        "Congratulations on your promotion!",
        "Let me know when you're free for a call.",
        "Just wanted to say hi! It's been a while.",
        "Good luck with your interview!",
        "Thanks for the recommendation!",
        "Do you need a ride to the airport?",
        "Hope your day is going well!",
        "Looking forward to our vacation next month.",
        "We should plan a road trip soon.",
        "Enjoy your weekend!",
        "Thanks for the invite! I'll be there.",
        "Let me know if you need any help moving.",
        "Hope to catch up with you soon.",
        "Have a safe flight!",
        "Take care and talk soon.",
    ],
}

df = pd.DataFrame(data)
print(df.head())  # Print first few rows


# %%
df.to_csv("spam.csv" , index="False")

# %%
df.head()

# %%
df.loc[0, "text"]

# %%
print(df.columns)

# %%
df["text"][0]

# %%
df["text"][1]

# %% [markdown]
# **Shape of the data**

# %%
print("Input has {} rows and {} columns".format(len(df),len(df.columns)))

# %% [markdown]
# **How many spam ham are there**

# %%
print("Out of {} rows, {} are spam, {} are ham".format(len(df), len(df[df["label"] == "spam"]), len(df[df["label"] == "ham"])))


# %% [markdown]
# **How much data is missing**

# %%
print("Number of null valuES {}".format(df["label"].isnull().sum()))
print("Number of null valuES {}".format(df["text"].isnull().sum()))

# %% [markdown]
# **PREPROCESSING THE DATA**

# %% [markdown]
# **Cleaning the text data is necessary is necessary to highlight the attributes that you are going to use in ML Algorithms**
# 
# **Steps include**
# 
# **Remove Punctuation**
# 
# **Tokenization**
# 
# **Remove stopwords**
# 
# **Lemmitization/Stemming**

# %%

import string

# %%
string.punctuation

# %%
def remove_punct(text):
    text_nonpunct="".join([char for char in text if char not in string.punctuation])
    return text_nonpunct

df["text_clean"]=df["text"].apply(lambda x:remove_punct(x))
df.head()

# %% [markdown]
# **Tokenization - splitting a string or sentence into list of words**

# %%
import re 

def tokenize(text):
    tokens=re.split('\W',text)
    return tokens

df["text_tokenized"]=df["text_clean"].apply(lambda x:tokenize(x.lower()))

df.head()



# %% [markdown]
# **Removing Stpwords- These are commonly used words like the,end,but,if that dont contribute much to the meaning of the sentence**

# %%
stopwords=nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_list):
    text=[word for word in tokenized_list if word not in stopwords]
    return text

df["text_nostop"]=df["text_tokenized"].apply(lambda x: remove_stopwords(x))

df.head()

# %%
import nltk
nltk.download('stopwords')


# %%
print(df.columns)

# %% [markdown]
# **Stemming- reducing infected or derived words to their stem or root**

# %%
ps=nltk.PorterStemmer()

def stemming(tokenized_text):
    if isinstance(tokenized_text,str):
        tokenized_text=tokenized_text.split()
    text=[ps.stem(word) for word in tokenized_text]
    return text
df["text_stemmed"]=df["text_nostop"].apply(lambda x:stemming(x))
df.head()

# %% [markdown]
# **Lemmitization - grouping together the infected type of words so that they can be analysed to a single term , identified by word lemma**
# 
# **For eg-type, typing, types are the forms of same lemma type**

# %%
import nltk
nltk.download('wordnet')

# %%
wn=nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
    text=[wn.lemmatize(word) for word in tokenized_text]
    return text
df["text_lemmatized"]=df["text_nostop"].apply(lambda x:lemmatizing(x))

df.head()

# %% [markdown]
# **Vectorization-encoding text as integers to create feature vectors. In other context it will take individual text messages and convert it to numeric vector that represents the text message**
# 
# **Count vectorization- Creates adocument term matrix where the entry of each cell will be the count of number of times that word occurred in the document**

# %%
from sklearn.feature_extraction.text import CountVectorizer

def clean_text(text):
    text="".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split("\W",text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text

count_vect= CountVectorizer(analyzer=clean_text)

x_count=count_vect.fit_transform(df["text"])

print(x_count.shape)

# %%
data_sample=df[0:20]

count_vect_sample=CountVectorizer(analyzer=clean_text)
x_count_sample=count_vect_sample.fit_transform(data_sample["text"])

print(x_count_sample.shape)

# %% [markdown]
# **Sparse Matrix - A matrix where most entries are 0**

# %%
x_count_sample

# %%
x_count_df=pd.DataFrame(x_count_sample.toarray())

x_count_df

# %%
import warnings
warnings.filterwarnings("ignore")

x_count_df.columns=count_vect_sample.get_feature_names_out()
x_count_df

# %% [markdown]
# **TD-IDF (Term Frequency,Inverse Document frequency)**

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect=TfidfVectorizer(analyzer=clean_text)
x_tfidf=tfidf_vect.fit_transform(df["text"])

print(x_tfidf.shape)

# %% [markdown]
# **Apply tfidf to smaller circle**

# %%
data_sample=df[0:20]

tfidf_vect_sample=TfidfVectorizer(analyzer=clean_text)
x_tfidf_sample=tfidf_vect_sample.fit_transform(data_sample["text"])

print(x_tfidf_sample.shape)

# %%
x_tfidf_df=pd.DataFrame(x_tfidf_sample.toarray())
x_tfidf_df.columns=tfidf_vect_sample.get_feature_names_out()
x_tfidf_df

# %% [markdown]
# # Feature Engineering Feature Creation

# %%
df=pd.read_csv("spam.csv",header=0)

df.drop(columns=["Unnamed: 0"],inplace=True)


df

# %%


# %%

df.head()

# %%
print(df.columns)

# %%
print(df.shape)

# %%
df.head()

# %% [markdown]
# **Create feature for text message length**

# %%
df["body_len"]=df["text"].apply(lambda x:len(x)-x.count(" "))

print(df[["label","text","body_len"]].head())

# %% [markdown]
# **Create feature for the % of the text that is punctuation**

# %%
def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100

df["punct%"]=df["text"].apply(lambda x:count_punct(x))

df.head()

# %%
import matplotlib.pyplot as plt
import numpy as np

bins=np.linspace(0,200,40)

plt.hist(df["body_len"],bins)
plt.title("Body length ditribution")
plt.show()

# %%
bins=np.linspace(0,50,40)

plt.hist(df["punct%"],bins)
plt.title("Punctuation and Distribution")
plt.show()

# %% [markdown]
# # Building Machine learning Classifiers using Random Forest Model

# %%
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# %%
df=pd.read_csv("spam.csv",header=0)

df.drop(columns=["Unnamed: 0"],inplace=True)


df.head()

# %%
def count_punct(text):
    count=sum(1 for char in text if char in string.punctuation)
    return round(count/(len(text)-text.count(" ")),3)*100

df["body_len"]=df["text"].apply(lambda x:len(x)-x.count(" "))


df["punct%"]=df["text"].apply(lambda x:count_punct(x))

df.head()

# %%
def clean_text(text):
    text="".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split("\W",text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect=TfidfVectorizer(analyzer=clean_text)
x_tfidf=tfidf_vect.fit_transform(df["text"])

# %%
x_features=pd.concat([df['body_len'],df["punct%"],pd.DataFrame(x_tfidf.toarray())],axis=1)
x_features.columns =x_features.columns.astype(str)
x_features

# %% [markdown]
# # Model using train test Split

# %%

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# %%
x_train, x_test, y_train, y_test = train_test_split(x_features, df["label"], test_size=0.3, random_state=0)

# %%
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=500,max_depth=20,n_jobs=-1)
rf_model=rf.fit(x_train,y_train)

# %%
print(x_train.dtypes)
print(x_train.head())

# %%
sorted(zip(rf_model.feature_importances_,x_train.columns),reverse=True)[0:10]

# %%
y_pred=rf_model.predict(x_test)

precision,recall,fscore,support=score(y_test,y_pred,pos_label="spam" ,average="binary")

# %%
print('precision {} / recall {} /Accuracy {}'.format(round(precision,3),
                                                        round(recall,3),
                                                        round(y_pred==y_test).sum()/len(y_pred),3))

# %%
print(rf_model)

# %%
import pandas as pd

def predict_message(msg, model, tfidf_vect):
    # Convert to DataFrame
    test_df = pd.DataFrame({'text': [msg]})
    
    # Feature Engineering (Apply same transformations as training)
    test_df['body_len'] = test_df['text'].apply(lambda x: len(x) - x.count(" "))
    test_df['punct%'] = test_df['text'].apply(lambda x: count_punct(x))  # Ensure count_punct() is defined

    # Convert text into TF-IDF features
    x_test_tfidf = tfidf_vect.transform(test_df['text']).toarray()
    x_test_features = pd.concat([test_df[['body_len', 'punct%']], pd.DataFrame(x_test_tfidf)], axis=1)
    x_test_features.columns = x_test_features.columns.astype(str)


    # Predict spam or ham
    prediction = model.predict(x_test_features)[0]
    return "Spam" if prediction == 1 else "Ham"

# Example Usage:
message = "Congratulations! You won a free lottery ticket. Claim now!"
print(predict_message(message, rf_model, tfidf_vect))


# %%
print(predict_message("Congratulations! You won a free lottery ticket. Claim now!", rf_model, tfidf_vect))


# %% [markdown]
# # to connect with streamlit

# %%
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset (Ensure it's properly preprocessed without clean_text)
import pandas as pd
df = pd.read_csv("spam.csv")  # Modify this if needed

# Separate features and labels
X = df["text"]  # Assuming "message" is the column with text
y = df["label"].map({"ham": 0, "spam": 1})  # Convert labels to 0 and 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Fix: Re-train Vectorizer without clean_text**
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the corrected vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Save the trained model
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved successfully!")


# %%
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()  # Define the vectorizer

# Assuming X_train is already cleaned
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)


# %%
import os

file_name = "spam_model.pkl"
file_path = os.path.abspath(file_name)

print("Absolute Path:", file_path)
print("File Exists:", os.path.exists(file_path))


# %%
import os
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load the trained model and vectorizer
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to predict if a message is spam or ham
def predict_message(msg, model, tfidf_vect):
    # Convert to DataFrame
    test_df = pd.DataFrame({'text': [msg]})
    
    # Feature Engineering (Apply same transformations as training)
    test_df['body_len'] = test_df['text'].apply(lambda x: len(x) - x.count(" "))
    test_df['punct%'] = test_df['text'].apply(lambda x: count_punct(x))

    # Convert text into TF-IDF features
    x_test_tfidf = tfidf_vect.transform(test_df['text']).toarray()
    x_test_features = pd.concat([test_df[['body_len', 'punct%']], pd.DataFrame(x_test_tfidf)], axis=1)
    x_test_features.columns = x_test_features.columns.astype(str)

    # Predict spam or ham
    prediction = model.predict(x_test_features)[0]
    return "Spam" if prediction == 1 else "Ham"

# Function to count punctuation percentage
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100

# Streamlit app
st.title("Spam/Ham Classification")

# Input text box for user to enter a message
user_input = st.text_input("Enter a message to classify:")

if user_input:
    prediction = predict_message(user_input, model, tfidf_vectorizer)
    st.write(f"The message is classified as: {prediction}")

# Display the first few rows of the dataset
file_path = "spam.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.write(df.head())
else:
    st.write("The file 'spam.csv' does not exist.")


# %%

