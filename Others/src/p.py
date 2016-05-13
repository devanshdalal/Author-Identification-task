import re,sys
import os,csv
import numpy as np
#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import svm,linear_model,ensemble
import pickle
import itertools
import string
from sklearn.linear_model import SGDClassifier
import subprocess
import random

#assigning predictor and target variables

trainX =[]
labels = []

testX =[]
test_labels = []

fractionTraining = 0.85

path = 'All/'
authors = os.listdir('All'); 
for auth in authors:  
	files = os.listdir(path+auth+'/');
	tmpX,tmpY=[],[]
	for file in files:
		f=open(path+auth+'/'+file, 'r')
		data = f.read().replace('\n', '')
		print path+auth+'/'+file, os.path.exists(path+auth+'/'+file),'size',len(data),auth   
		tmpX.append(data)
		tmpY.append(auth)
		f.close() 
	random.shuffle(tmpX)
	random.shuffle(tmpY)

	trainX=trainX+tmpX[:int(fractionTraining*len(tmpX)) ]
	labels=labels+tmpY[:int(fractionTraining*len(tmpY)) ]

	testX=testX+tmpX[int(fractionTraining*len(tmpY)):]
	test_labels=test_labels+tmpY[int(fractionTraining*len(tmpY)):]

# exit(0)

logfile = open('dump.txt','wb')

# tweets = [processTweet(t) for t in tweets];


######################################### LOGISTIC REGRESSION ########################################################

# vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1,stop_words='english',lowercase=False)
vectorizer = TfidfVectorizer( ngram_range=(1, 2), min_df=1,stop_words='english',lowercase=False) #,stop_words=stopwords)
# vectorizer = TfidfVectorizer(  min_df=1) #,stop_words=stopwords)

train_vectors = vectorizer.fit_transform(trainX)
test_vectors = vectorizer.transform(testX)

logreg = SGDClassifier( loss='log', alpha=0.000001, penalty='l2' , n_iter=5, shuffle=True);
# logreg = linear_model.LogisticRegression(solver='newton-cg')
logreg.fit(train_vectors, labels)

# f = open("GOLD.txt","wb")
# f.write("\n".join([str(x) for x in test_labels]) )
# f.close()

Z = logreg.predict(test_vectors)

print 'accuracy',logreg.score(test_vectors,test_labels)
print 'accuracy_native',np.average(np.array(Z)==np.array(test_labels))
# for i,x in enumerate(Z):
# 	if x==2 and random.random()<0.5:
# 		Z[i]=0;

# f = open("MY.txt","wb")
# f.write("\n".join([str(x) for x in Z]) )
# f.close()

# print subprocess.check_output( 'python fscore.py GOLD.txt MY.txt' ,shell=True)

logfile.close()

# Saving the objects:
# with open('objs.pickle'+sys.argv[2], 'w') as f:
#     pickle.dump([vectorizer,logreg], f)