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
import nltk
import nltk.data

#assigning predictor and target variables

trainX =np.array([])
labels = []

testX =np.array([])
test_labels = []
extra_features_train=[] # n X p   // p=10
extra_features_test=[] # n X p   // p=10

fractionTraining = 0.80
#  TODO: change Q to array
Q = [100.0, 25.0, 25.0, 50.0, 25.0, 50.0, 100.0, 25.0, 50.0, 25.0];
#Q = 20.0 

################################################## Extra_features ###############################################################

"""
#returns the number of sentences in text
def count_sents(text):
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle');
	sents = sent_detector.tokenize(text);

	num_sents = len(sents);
	return num_sents;
"""

#returns the number of sentences in text
def count_sents(text):
	sents = nltk.sent_tokenize(text);
	num_sents = len(sents);
	return num_sents;

#returns the number of tokens in text
def count_tokens(text):
	tokens = nltk.word_tokenize(text);
	num_tokens = len(tokens);
	return num_tokens;

#returns the number of tokens without punctuations in text
def count_tokens_wop(text):
	tokens_wop = nltk.word_tokenize(text.translate(None, string.punctuation));
	num_tokens_wop = len(tokens_wop);
	return num_tokens_wop;

#returns fraction of tokens that are punctuations
def frac_puncs(text):
	num_tokens = count_tokens(text);
	num_tokens_wop = count_tokens_wop(text);
	return (float(num_tokens - num_tokens_wop))/(float(num_tokens)); 

#returns average token length
def avg_token_len(text):
	tokens = nltk.word_tokenize(text);
	total_len=0;
	for token in tokens:
		total_len = total_len + len(token);
	return (float(total_len))/(float(len(tokens)));

#returns average sentence length (in terms of words)
def avg_sent_len1(text):
	sents = nltk.sent_tokenize(text);
	total_len=0;
	for sent in sents:
		total_len = total_len + count_tokens(sent);
	return (float(total_len))/(float(len(sents)));

#returns average sentence length (in terms of chars)
def avg_sent_len2(text):
	sents = nltk.sent_tokenize(text);
	total_len=0;
	for sent in sents:
		total_len = total_len + len(sent);
	return (float(total_len))/(float(len(sents)));	


#returns standard deviation of lengths of tokens
def stdev_token_len(text):
	tokens = nltk.word_tokenize(text);
	len_array = [];
	for token in tokens:
		len_array.append(len(token));
	return np.std(np.array(len_array));

#returns standard deviation of lengths of sentences (wrt words)
def stdev_sent_len1(text):
	sents = nltk.sent_tokenize(text);
	len_array = [];
	for sent in sents:
		len_array.append(count_tokens(sent));
	return np.std(np.array(len_array));


#returns standard deviation of lengths of sentences (wrt chars)
def stdev_sent_len2(text):
	sents = nltk.sent_tokenize(text);
	len_array = [];
	for sent in sents:
		len_array.append(len(sent));
	return np.std(np.array(len_array));

#returns number of digits in the text
def num_digits(text):
	ans=0;
	for c in text:
		if(c.isdigit()):
			ans = ans+1;
	return ans;

#returns the array with additional feature values
def extra_feats(text):
	feat_array=[];
	feat_array.append(count_tokens(text));			#number of tokens
	feat_array.append(count_sents(text));			#number of sentences
	feat_array.append(frac_puncs(text));			#fraction of punctuations
	feat_array.append(avg_token_len(text));			#average token length
	feat_array.append(stdev_token_len(text));		#standard deviation of token lengths
	feat_array.append(avg_sent_len1(text));			#average sentence length (wrt words)
	feat_array.append(avg_sent_len2(text));			#average sentence length (wrt chars)
	feat_array.append(stdev_sent_len1(text));		#standard deviation of sentence lengths (wrt words)
	feat_array.append(stdev_sent_len2(text));		#standard deviation of sentence lengths (wrt chars)
	feat_array.append(num_digits(text));			#number of digits
	return feat_array;

####################################################################################################################

path = 'C50train/';
authors = os.listdir(path); 
for auth in authors:  
	files = os.listdir(path+auth+'/');
	tmpX,tmpY=np.array([]),[]
	for file in files:
		f=open(path+auth+'/'+file, 'r')
		data = f.read().replace('\n', ' ')
		# print path+auth+'/'+file, os.path.exists(path+auth+'/'+file),'size',len(data),auth   
		tmpX=np.append(tmpX,data)
		tmpY=tmpY+[auth]
		f.close() 
	random.shuffle(tmpX)

	part_for_traning=tmpX[:int(fractionTraining*len(tmpX))];
	for x in xrange(part_for_traning.shape[0]):
		extra_features_train.append( extra_feats( part_for_traning[x] ) );
	
	# print part_for_traning
	trainX=np.append(trainX, part_for_traning)
	labels=labels+tmpY[:int(fractionTraining*len(tmpY)) ]

	part_for_testing=tmpX[int(fractionTraining*len(tmpX)):];
	for x in xrange(part_for_testing.shape[0]):
		extra_features_test = extra_features_test +  [extra_feats( part_for_testing[x] )] ;

	testX=np.append(testX,part_for_testing)
	test_labels=test_labels+tmpY[int(fractionTraining*len(tmpY)):]

# exit(0)

logfile = open('dump.txt','wb')

# tweets = [processTweet(t) for t in tweets];


######################################### LOGISTIC REGRESSION ########################################################

# print extra_features_train

# vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1,stop_words='english',lowercase=False)
vectorizer = TfidfVectorizer( ngram_range=(1, 2),stop_words='english',lowercase=False) #,stop_words=stopwords) # 
# vectorizer = TfidfVectorizer( ngram_range=(1, 2), min_df=1,stop_words='english',lowercase=False) #,stop_words=stopwords)
# vectorizer = TfidfVectorizer(  min_df=1) #,stop_words=stopwords)

# print trainX
native_train_vectors = vectorizer.fit_transform(trainX)
native_test_vectors = vectorizer.transform(testX)

vectorizer = TfidfVectorizer( ngram_range=(1, 2),stop_words='english',lowercase=False) #,stop_words=stopwords) # 

extra_features_train = np.array(extra_features_train)
extra_features_test = np.array(extra_features_test)

upper_bound,lower_bound = np.amax(extra_features_train,0),np.amin(extra_features_train,0)
print upper_bound, lower_bound
for x in xrange(trainX.shape[0]):
	for y in xrange(extra_features_train.shape[1]):
		# print 'hhhh', extra_features_train[x][y]
		bucket = int(Q[y]*(extra_features_train[x][y]-lower_bound[y])/(upper_bound[y]-lower_bound[y]))
		trainX[x]=trainX[x] +" F"+ str(y) +"_"+ str(bucket) 

upper_bound,lower_bound = np.amax(extra_features_test,0),np.amin(extra_features_test,0)
for x in xrange(testX.shape[0]):
	for y in xrange(extra_features_test.shape[1]):
		bucket = int(Q[y]*(extra_features_test[x][y]-lower_bound[y])/(upper_bound[y]-lower_bound[y]))
		testX[x]=testX[x] +" F"+ str(y) +"_"+ str(bucket) 

train_vectors = vectorizer.fit_transform(trainX)
test_vectors = vectorizer.transform(testX)

logreg = SGDClassifier( loss='log', alpha=0.000001, penalty='l2' , n_iter=5, shuffle=True);
# logreg = linear_model.LogisticRegression(solver='newton-cg')

# np.concatenate( (train_vectors.todense(),extra_features) ,1) 

logreg.fit(train_vectors, labels)

# f = open("GOLD.txt","wb")
# f.write("\n".join([str(x) for x in test_labels]) )
# f.close()

Z = logreg.predict(test_vectors)

# print 'accuracy_with_features',logreg.score(test_vectors,test_labels)
print 'accuracy_with_features',np.average(np.array(Z)==np.array(test_labels))
# for i,x in enumerate(Z):
# 	if x==2 and random.random()<0.5:
# 		Z[i]=0;

# f = open("MY.txt","wb")
# f.write("\n".join([str(x) for x in Z]) )
# f.close()
logreg = SGDClassifier( loss='log', alpha=0.000001, penalty='l2' , n_iter=5, shuffle=True);
logreg.fit(native_train_vectors, labels)
Z = logreg.predict(native_test_vectors)

# print 'accuracy_native',logreg.score(native_test_vectors,test_labels)
print 'accuracy_native',np.average(np.array(Z)==np.array(test_labels))
# print subprocess.check_output( 'python fscore.py GOLD.txt MY.txt' ,shell=True)

logfile.close()

# Saving the objects:
# with open('objs.pickle'+sys.argv[2], 'w') as f:
#     pickle.dump([vectorizer,logreg], f)