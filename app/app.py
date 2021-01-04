import nltk
import pickle
from flask import Flask, render_template, request
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer   
 
from nltk.stem.porter import PorterStemmer 
import re

from Machine_Learning import *



filename = 'machine_model.pkl'
regressor = pickle.load(open(filename, 'rb'))
#nltk.download('vader_lexicon')
#nltk.download('punkt')

#from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
		
	if request.method == 'POST':

		review_text = request.form.get('text')

		x = get_conv(review_text)

		vec = cv.transform([x])

		output = classifier.predict(vec)


		return render_template('index.html', result_review = output)
		




'''
		empty_space = []

		

		review = re.sub('[^a-zA-Z]', ' ', review_text)

		review = review_text.lower()

		review = review_text.split()

		ps = PorterStemmer()

		review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

		review = ' '.join(review)

		
		
		sid = SentimentIntensityAnalyzer()
		result = sid.polarity_scores(review)

		if result['compound'] < 0:
			polarity_scores='Negative'
			return render_template('index.html', result_review = polarity_scores)

		elif result['compound'] > 0:
			polarity_scores='Positive'
			return render_template('index.html', result_review = polarity_scores)

		else:
			polarity_scores='Neutral'	

			return render_template('index.html', result_review = polarity_scores) '''

    

    	
if __name__ == '__main__':
	app.run(debug=True)



