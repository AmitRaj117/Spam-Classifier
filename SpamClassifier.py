import pandas as pd
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
print(messages)
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

corpus = []
for i in range (len (messages)):
    review = re.sub( '[^a-zA-Z]',' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
vc = TfidfVectorizer()
x = vc.fit_transform(corpus).toarray()
print(x)

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.20)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train , y_train)
y_pred = spam_detect_model.predict( x_test)

from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)
print(confusion_m)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)