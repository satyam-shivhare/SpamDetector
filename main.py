import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import chardet
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import seaborn as sns
from wordcloud import WordCloud
from collections import  Counter
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# this code finds the encoding for the file we are tring to open with read_csv() function. Here is the result on my system : {'encoding': 'Windows-1252', 'confidence': 0.7272080023536335, 'language': ''}
with open('spam.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)


df = pd.read_csv('spam.csv', encoding='Windows-1252')
df.sample(5)
df.shape
# steps in the project :
# 1. DATA CLEANING
# 2. Exploratory Data Analysis
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvememt
# 7. Website Creation
# 8. Deployment


 # ==================   STEP 1 - DATA CLEANING ==================

df.info

# we are going to drop last 3 columns as these columns have mostly null or NaN values
df.drop(columns=['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'], inplace=True)

# renaming the remaining columns for better readability

df.rename(columns={'v1':'target', 'v2':'text'},inplace=True)

# renaming the fields 'ham' and 'spam' to number using label encoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# checking if there are any more missing values
#df.isnull().sum()
print(df.isnull().sum())

# checking for duplicated values
print(df.duplicated().sum()) # we actually get 403 duplicated values

#removing the duplicates now

df = df.drop_duplicates(keep='first')

print(df.duplicated().sum()) # now we don't get any duplicates


#============================  STEP 2 : Exploratory Data Analysis ===========================

# figuring out how many spams and hams we have in our data
print(df['target'].value_counts())

#plotting a graph for the number of hams and spams using matplotlib

plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct='%0.2f')
plt.show()
# here we see that roughly we have 87% hams and 12% spams.
# ========= that means data is imbalanced =========

# now we will find number of alphabets , words and sentences in our sms messages and perform some analysis on them

# we need some dependencies of nltk;
nltk.download('punkt')

# so no we are making 3 columns for analysis :
# a) number of character in sms message.
# b) number of words in sentences.
# c) number of sentences in sms message.

# number of characters in each message
df['num_characters'] = df['text'].apply(len)

# number of words in each message. We will need to use nltk
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

# number of sentences in each message. We will need to use nltk
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

# now have the 3 new coloumns which can be called new features too.

# now we will see some statistic measurements like mean, min , max , percentiles etc on these new features
print(df[['num_characters','num_words','num_sentences']].describe())

# we can also do the analysis for ham and spam separately

# this is for 'ham' messages
print(df[df['target']==0][['num_characters','num_words','num_sentences']].describe())

# this is for 'spam' messages
print(df[df['target']==1][['num_characters','num_words','num_sentences']].describe())


# now we will plot histograms for both spam and ham categories using seaborn
plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'], color='red')
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'], color='red')
plt.show()

sns.pairplot(df, hue='target')
plt.show()

# we can also see correlations and put them in heat map
sns.heatmap(df.corr(),annot=True)
plt.show()


#============================== STEP 3 - DATA PREPROCESSING ===========================

# since we have text as our data we are going to do the below preprocessing steps

# 1.) Lowercase
# 2.) Tokenization
# 3.) Removing special characters
# 4.) Removing stop words and punctuations
# 5.) Stemming


# we are creating a function that will perform all of the above steps in one go

nltk.download('stopwords') # it is a necessary dependency
def transform_text(text):
    # lowercase
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters
    y = []
    for i in text:
        if(i.isalnum() or i.isalpha()):
            y.append(i)

    text = y[:]
    y.clear()

    # removing stop words and punctuations
    # we did it outside function : nltk.download('stopwords')
    for i in text:
        if i not in stopwords.words('English') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # stemming
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    text = y[:]
    y.clear()
    return " ".join(text)


# now we will make a new column called transform_text
df['transform_text'] = df['text'].apply(transform_text)
print(df.head())

# now we will make a word cloud that will help us to identify which words are usually used in hams and spams

# for spam messages
wc = WordCloud(width=500, height=500, min_font_size=10,background_color='white')
spam_wc = wc.generate(df[df['target']==1]['transform_text'].str.cat(sep=" "))
plt.figure(figsize=(12,6))
plt.imshow(spam_wc)
plt.show()

# the bigger the word in the graph the more it is used accross the data

# for ham messages
ham_wc = wc.generate(df[df['target']==0]['transform_text'].str.cat(sep=" "))
plt.figure(figsize=(12,6))
plt.imshow(ham_wc)
plt.show()


# now we are going to figure our top 50 words used in the spam and ham messages

# for spam
spam_corpus = []
for msg in df[df['target']==1]['transform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

#for ham
ham_corpus = []
for msg in df[df['target']==0]['transform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


#================================== STEP 4 - Machine learning Modelling ==========================

# ====================== First Model - Naive Bayes =================

# so in order to run any algo in text we first need to vetorize our text. Basically we are just converting text to numbers

# we are going to first use the 'bag of words' method for vectorization
cv = CountVectorizer()

X = cv.fit_transform(df['transform_text']).toarray()

print(X.shape)

y = df['target'].values

# now we can apply train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# now we will be using 3 different models of naive bayes and see which one gives best output
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# now we will train model and then do prediction and them measure different metrics

# for gnb model
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

# for mnb model
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

# for bnb model
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

# ====================  on running the code till above lines, we can conclude that bernoulliNB is performing the best but now we will try different methods like TFid vectorization to see if we can improve the result ========================

tfidf = TfidfVectorizer(max_features=3000) # i am doing max features to 3000 for improving the model after I chose the naive bayes model
X_tfidf  = tfidf.fit_transform(df['transform_text']).toarray()

# below commented to code was used to see if the scaling of X would help us to improve our model or not.
# It was not a good idea because accuracy got better slightly but precision went down.

#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# appending the num_character col to X. It also did not help in improving our model
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))

print(X_tfidf.shape)

y = df['target'].values

# now we can apply train-test split
X_tfidf_train,X_tfidf_test,y_train,y_test = train_test_split(X_tfidf,y,test_size=0.2,random_state=2)

# now we will be using 3 different models of naive bayes and see which one gives best output
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# now we will train model and then do prediction and them measure different metrics

# for gnb model
gnb.fit(X_tfidf_train,y_train)
y_pred1_tfidf = gnb.predict(X_tfidf_test)
print(accuracy_score(y_test, y_pred1_tfidf))
print(confusion_matrix(y_test, y_pred1_tfidf))
print(precision_score(y_test, y_pred1_tfidf))

# for mnb model
mnb.fit(X_tfidf_train,y_train)
y_pred2_tfidf= mnb.predict(X_tfidf_test)
print(accuracy_score(y_test, y_pred2_tfidf))
print(confusion_matrix(y_test, y_pred2_tfidf))
print(precision_score(y_test, y_pred2_tfidf))

# for bnb model
bnb.fit(X_tfidf_train,y_train)
y_pred3_tfidf = bnb.predict(X_tfidf_test)
print(accuracy_score(y_test, y_pred3_tfidf))
print(confusion_matrix(y_test, y_pred3_tfidf))
print(precision_score(y_test, y_pred3_tfidf))



# after performing above tests we can see that the 'Precision_score' of the mnb is highest and since we have imbalanced data, the precision score is more important that accuracy
# so now we have options to go with bernoulliNB or multinomialNB. Since the Precision_score matter more in our case. We should go with multinomialNB.
# =================  out of available options we choose MULTINOMIAL NAIVE BAYES ====================


# Now we will test some more models and see their performances
# I am writing import statements here so that you can see which models we are going to use for our analysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier



svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, precision


train_classifier(svc, X_tfidf_train, y_train, X_tfidf_test, y_test)
# here svc model is also very good as it yields high acuracy and precision on running program. Let us se what happens when we use other models

# we will store the accuracy and precision scores of different models and then see which ones are best to use for our project
accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_tfidf_train, y_train, X_tfidf_test, y_test)

    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

# now we are storing the accuracy_scores and precision_scores and making a dataframe of them so that we can perform different operations
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

# below variable store the above dataframe in the decreasing order of accuracy_score
performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")

# you can also plot graphs for better visualization
sns.catplot(x = 'Algorithm', y='value',hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# ==============================  now we are going to focus on improving our model ======================
# we are choosing the Naive Bayes because it is the model that is giving best predictions

# we are going to use some methods to make improvements

# 1.) Change the max_features parameters of TfIdf

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)

new_df = performance_df.merge(temp_df,on='Algorithm')

# 2.) scaling the X_tfidf

# below scaling actually increased the accuracy slightly but it messed up the precision and also there were no significant changes in the output of other models.
new_df_scaled = new_df.merge(temp_df,on='Algorithm')

# 3.) We created 3 new features and added them to our original dataframe df. I will now append the num_characters in our input and see if our model performance improves
# but it also did not gave use any good improvement to us.
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)
new_df_scaled.merge(temp_df,on='Algorithm')


# 4.) Voting Classifier. It also did not gave us ay much improvement that our best bet.
# svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
# mnb = MultinomialNB()
# etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
#
# from sklearn.ensemble import VotingClassifier
#
# voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
# voting.fit(X_tfidf_train,y_train)
#
# y_pred2_tfidf = voting.predict(X_test)
# print("Accuracy",accuracy_score(y_test,y_pred2_tfidf))
# print("Precision",precision_score(y_test,y_pred2_tfidf))


# 5.) Applying stacking. This also did Not give us any great improvement.
# estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
# final_estimator=RandomForestClassifier()
#
# from sklearn.ensemble import StackingClassifier
#
# clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
#
# clf.fit(X_tfidf_train,y_train)
# y_pred = clf.predict(X_tfidf_test)
# print("Accuracy",accuracy_score(y_test,y_pred2_tfidf))
# print("Precision",precision_score(y_test,y_pred2_tfidf))

# ========================== so now we are just going to use the multinomial naive bayes because it is the best performing till now ==============
# so now that we have decided our model, we will create a pipeline and then convert that pipeline into a website

#=== Steps in pipeline : ======
# 1.) Preprocess the email or sms
# 2.) Vectorize the text
# 3.) Apply the algorithm

# now we are going to pickle two files from here
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))

