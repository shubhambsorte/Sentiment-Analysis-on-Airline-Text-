
import pandas as pd


import re  
import nltk  

from nltk.corpus import stopwords 
  
 
from nltk.stem.porter import PorterStemmer 

import pickle



def get_conv(x):
    
         
    
        x = re.sub('[^a-zA-Z]', ' ', x)   
        x = x.lower() 
        
        x = x.split() 
        
        ps = PorterStemmer()  
      
        x = [ps.stem(word) for word in x 
                if not word in set(stopwords.words('english'))] 
        
    
        x = ' '.join(x)
        
        return x
         
        

df = pd.read_csv('airline.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)

df['text'] = df['text'].apply(lambda x: get_conv(x))  


from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()  

X = cv.fit_transform(df['text'])

y = df['airline_sentiment']

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

#print(model.score(X_train, y_train))

filename = 'machine_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

