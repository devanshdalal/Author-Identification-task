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
from collections import Counter
import porter
import fingerprintgenerator 
import pyintertextuality as itx

#assigning predictor and target variables
trainX =np.array([])
labels = []

testX =np.array([])
test_labels = []
extra_features_train=[] # n X p   // p=5
extra_features_test=[] # n X p   // p=5

features_are_on=False
fingerprints_on = True
fractionTraining = 0.50
#  TODO: change Q to array
Q = [100.0, 25.0, 25.0, 50.0, 25.0, 50.0, 100.0, 25.0, 50.0, 25.0];
original_author_list = ['SamuelPerry','MartinWolk','HeatherScoffield','ScottHillis','KarlPenhaul','JaneMacartney','JimGilchrist','AlanCrosby','LynnleyBrowning','AlexanderSmith','SarahDavison','TimFarrand','MarkBendeich','KouroshKarimkhany','EricAuchard','TanEeLyn','DarrenSchuettler','JoeOrtiz','DavidLawder','JoWinterbottom','JanLopatka','RobinSidel','ToddNissen','GrahamEarnshaw','SimonCowell','PeterHumphrey','TheresePoletti','PatriciaCommins','BenjaminKangLim','MarcelMichelson','JonathanBirt','PierreTran','KirstinRidley','MureDickie','BernardHickey','LydiaZajc','NickLouth','KevinMorrison','JohnMastrini','KevinDrawbaugh',"LynneO'Donnell",'BradDorfman','FumikoFujisaki','MatthewBunce','WilliamKazer','MichaelConnor','RogerFillion','EdnaFernandes','KeithWeir','AaronPressman'][:10]
author_indx={}
author_text={}

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
"""
#returns standard deviation of lengths of sentences (wrt words)
def stdev_sent_len1(text):
	tokens = nltk.word_tokenize(text);
	len_array = [];
	for token in tokens:
		len_array.append(len(token));
	return np.std(np.array(len_array));
"""

#returns standard deviation of lengths of tokens
def stdev_token_len(text):
	tokens = nltk.word_tokenize(text);
	len_array = [];
	for token in tokens:
		len_array.append(len(token));
	return np.std(np.array(len_array));

#returns number of digits in the text
def num_digits(text):
	ans=0;
	for c in text:
		if(c.isdigit()):
			ans = ans+1;
	return ans;

def similarity(text):
	# hash using winnowing algorithm
	currmax=-1;
	res = ''
	print >>sys.stderr , len(text)
	for i,x in enumerate(author_text):
		winnow1 = itx.algorithms.winnow(text[:2000])
		winnow2 = itx.algorithms.winnow(author_text[x][:2000])
		# look for matches of at least 'threshold' hashes
		a = itx.compare_fingerprints(winnow1, winnow2, threshold=3)
		if len(a)>currmax:
			currmax=len(a);
			res=x;
	return res

#returns the array with additional feature values
def extra_feats(text):
	feat_array=[];
	if(features_are_on):
		feat_array.append(count_tokens(text));			#number of tokens
		# feat_array.append(count_sents(text));			#number of sentences ??? Not a good feature
		feat_array.append(frac_puncs(text));			#fraction of punctuations
		# feat_array.append(avg_token_len(text));		#average token length ??? BAD feature
		feat_array.append(stdev_token_len(text));		#standard deviation of token lengths
		feat_array.append(avg_sent_len1(text));			#average sentence length (wrt words)
		feat_array.append(avg_sent_len2(text));			#average sentence length (wrt chars)
		# feat_array.append(num_digits(text));			#number of digits ???? BAD FEATURE
	return feat_array;

####################################################################################################################
# path = 'All/'
# authors = os.listdir(path); 
# for auth in authors:  
# 	original_author_list.append(auth)
# 	files = os.listdir(path+auth+'/');
# 	tmpX,tmpY=np.array([]),[]
# 	for file in files:
# 		f=open(path+auth+'/'+file, 'r')
# 		data = f.read().replace('\n', '')
# 		# print path+auth+'/'+file, os.path.exists(path+auth+'/'+file),'size',len(data),auth   
# 		tmpX=np.append(tmpX,data)
# 		tmpY=tmpY+[auth]
# 		f.close() 
# 	# tmpX=np.array( sorted(list(tmpX),key = len, reverse=True) )
# 	# random.shuffle(tmpX)

# 	part_for_traning=tmpX[:int(fractionTraining*len(tmpX))];
# 	for x in xrange(part_for_traning.shape[0]):
# 		extra_features_train.append( extra_feats( part_for_traning[x] ) );
	
# 	# print part_for_traning
# 	trainX=np.append(trainX, part_for_traning)
# 	labels=labels+tmpY[:int(fractionTraining*len(tmpY)) ]

# 	part_for_testing=tmpX[int(fractionTraining*len(tmpX)):];
# 	for x in xrange(part_for_testing.shape[0]):
# 		extra_features_test = extra_features_test +  [extra_feats( part_for_testing[x] )] ;

# 	testX=np.append(testX,part_for_testing)
# 	test_labels=test_labels+tmpY[int(fractionTraining*len(tmpY)):]

#-------------------------------------------------------------------------------------

path1 = 'training/'
authors = os.listdir(path1)[:10]; 
for auth in authors:
	files = os.listdir(path1+auth+'/');
	tmpX,tmpY=np.array([]),[]
	for file in files:
		f=open(path1+auth+'/'+file, 'r')
		data = f.read().replace('\n', ' ')
		# data=porter.porter_stem(data)
		# print path+auth+'/'+file, os.path.exists(path+auth+'/'+file),'size',len(data),auth   
		tmpX=np.append(tmpX,data)
		tmpY=tmpY+[auth]
		f.close()
	# tmpX=np.array( sorted(list(tmpX),key = len, reverse=True) )
	# random.shuffle(tmpX)

	part_for_traning=tmpX
	for x in xrange(part_for_traning.shape[0]):
		extra_features_train.append( extra_feats( part_for_traning[x] ) );
		if fingerprints_on:
			fpg = fingerprintgenerator.FingerprintGenerator(input_string=tmpX[x])
			fpg.generate_fingerprints()
			fpg.fingerprints=list(set([y[0] for y in fpg.fingerprints]))
			for y in fpg.fingerprints:
				tmpX[x]+=' '+str(y)+'L'
	# print part_for_traning
	print >> sys.stderr ,'TRAINING',auth
	trainX=np.append(trainX, tmpX)
	labels=labels+tmpY 

for i,x in enumerate(trainX):
	if labels[i] not in author_text:
		author_text[ labels[i] ]=''
	author_text[ labels[i] ]+=' '+x

# exit(0)
# for i,x in enumerate(trainX):
# 	trainX[i]+=' '+similarity(trainX[i]);
# print sys.stderr , 'SIMILARITY Done'

path2 = 'testing/'
authors = os.listdir(path2)[:10]; 
for auth in authors:  
	# original_author_list.append(auth)
	files = os.listdir(path2+auth+'/');
	tmpX,tmpY=np.array([]),[]
	for file in files:
		f=open(path2+auth+'/'+file, 'r')
		data = f.read().replace('\n', ' ')
		# data=porter.porter_stem(data)
		# print path+auth+'/'+file, os.path.exists(path+auth+'/'+file),'size',len(data),auth   
		tmpX=np.append(tmpX,data)
		tmpY=tmpY+[auth]
		f.close() 
	# tmpX=np.array( sorted(list(tmpX),key = len, reverse=True) )
	# random.shuffle(tmpX)

	for x in xrange(tmpX.shape[0]):
		extra_features_test = extra_features_test +  [extra_feats( tmpX[x] )] ;
		if fingerprints_on:
			fpg = fingerprintgenerator.FingerprintGenerator(input_string=tmpX[x])
			fpg.generate_fingerprints()
			fpg.fingerprints=list(set([y[0] for y in fpg.fingerprints]))
			for y in fpg.fingerprints:
				tmpX[x]+=' '+str(y)+'L'
		# tmpX[x]+=' '+similarity(tmpX[x]);
	print >> sys.stderr ,'TESTING',auth
	testX=np.append(testX,tmpX)
	test_labels=test_labels+tmpY

#----------------------------------------------------------------------------------

for i,x in enumerate(original_author_list):
	author_indx[x]=i+1;

logfile = open('dump.txt','wb')
# exit(0)

######################################### Error analysis ########################################################

def print_grid(grid):
	print '  |',
	for x in xrange(len(grid)):
		print '{0:2d}'.format(x+1),
	print
	print '--|',
	for x in xrange(len(grid)):
		print '--',
	print
	for id,row in enumerate(grid):
		print '{0:2d}|'.format(id+1),
		for e in row:
			print '{0:2d}'.format(e),
		print

def calc_scores(arr1, arr2, auth_list):
	num_authors = len(auth_list);
	num_docs = len(arr1);
	confus_mat = [];
	
	for i in range(num_authors):
		row = [];
		for j in range(num_authors):
			row.append(0);
		confus_mat.append(row);

	for i in range(num_docs):
		row_num = auth_list.index(arr1[i]);
		col_num = auth_list.index(arr2[i]);
		confus_mat[row_num][col_num] = confus_mat[row_num][col_num]+1;

	correct_preds=0;
	for i in range(num_authors):
		correct_preds = correct_preds + confus_mat[i][i];
	accuracy = (float(correct_preds))/(float(num_docs));

	class_precs = [];
	for i in range(num_authors):
		total_class_preds=0;
		for j in range(num_authors):
			total_class_preds = total_class_preds + confus_mat[j][i];
		if total_class_preds==0:
			print "TOTAL_CLASS_PREDICTION is ZERO",total_class_preds
		class_prec = (float(confus_mat[i][i]))/(float(total_class_preds));
		class_precs.append(class_prec);

	class_recalls = [];
	for i in range(num_authors):
		total_class_docs=0;
		for j in range(num_authors):
			total_class_docs = total_class_docs + confus_mat[i][j];
		class_recall = (float(confus_mat[i][i]))/(float(total_class_docs));
		class_recalls.append(class_recall);

	avg_macro_prec = sum(class_precs)/(float(num_authors));
	avg_macro_recall = sum(class_recalls)/(float(num_authors));
	avg_macro_fmeasure = (2.0 * avg_macro_prec * avg_macro_recall)/(avg_macro_prec + avg_macro_recall);

	print "############# Confusion Matrix #############";
	print_grid(confus_mat)
	print "accuracy = " ,accuracy;
	print "average macro precision = ", avg_macro_prec;
	print "average macro recall = " ,avg_macro_recall;
	print "average macro fmeasure = " , avg_macro_fmeasure;


ngrams=(1, 2)
loss_function='log'
Random_Seed=None
shuffleOn= False

def new_vectorizer():
	return TfidfVectorizer( ngram_range=ngrams,stop_words='english',lowercase=False) #,stop_words=stopwords) #

def new_classifier():
	return SGDClassifier( loss=loss_function, alpha=0.000001, penalty='l2' , n_iter=5, shuffle=shuffleOn,random_state=Random_Seed);


######################################### LOGISTIC REGRESSION ########################################################

print trainX[0],testX[0]
print labels[0],test_labels[0]

# vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1,stop_words='english',lowercase=False)
vectorizer = new_vectorizer() 
# vectorizer = TfidfVectorizer( ngram_range=(1, 2), min_df=1,stop_words='english',lowercase=False) #,stop_words=stopwords)

# print trainX
native_train_vectors = vectorizer.fit_transform(trainX)
native_test_vectors = vectorizer.transform(testX)


extra_features_train = np.array(extra_features_train)
extra_features_test = np.array(extra_features_test)

upper_bound,lower_bound = np.amax(extra_features_train,0),np.amin(extra_features_train,0)
print upper_bound
print lower_bound
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

######################################### resolving confusion ########################################################
confusion_list=[] #[(3,17),(16,26),(6,29),(4,34),(34,45),(21,39)]
confused_authors={}
for i,x in enumerate(confusion_list):
	confused_authors[x[0]]=confused_authors[x[1]]=i;

models=[]
vectorizers=[]

for tup in confusion_list:
	tmpVectorizer=new_vectorizer();
	tmpClassifier=new_classifier();
	special_train_examples=[]
	special_train_labels=[]
	for x in xrange(trainX.shape[0]):
		if author_indx[labels[x]]==tup[0] or author_indx[labels[x]]==tup[1]:
			special_train_examples.append(trainX[x])
			special_train_labels.append(labels[x])
	special_train_vectors = tmpVectorizer.fit_transform(special_train_examples)

	tmpClassifier.fit(special_train_vectors, special_train_labels)
	vectorizers+=[tmpVectorizer]
	models+=[tmpClassifier]

def fine_tune(arg_train,arg_labels):
	for i,x in enumerate(arg_labels):
		if author_indx[x] in confused_authors:
			model_number = confused_authors[author_indx[x]]
			arg_train_vectors=vectorizers[model_number].transform( np.array([ arg_train[i] ]) )
			arg_labels[i]=  models[model_number].predict(arg_train_vectors)[0] # similarity(arg_train[i]) 
	return arg_labels
######################################################################################################################################

vectorizer = new_vectorizer()
train_vectors = vectorizer.fit_transform(trainX)
test_vectors = vectorizer.transform(testX)

logreg = new_classifier()
# logreg = linear_model.LogisticRegression(solver='newton-cg')
# np.concatenate( (train_vectors.todense(),extra_features) ,1) 

logreg.fit(train_vectors, labels)

# f = open("GOLD.txt","wb")
# f.write("\n".join([str(x) for x in test_labels]) )
# f.close()

Z = logreg.predict(test_vectors)
calc_scores(test_labels,Z,original_author_list)
Z=fine_tune(testX,Z)
calc_scores(test_labels,Z,original_author_list)

################################################################## Native scores ###########################################################

# logreg = new_classifier()
# logreg.fit(native_train_vectors, labels)
# Z = logreg.predict(native_test_vectors)
# calc_scores(test_labels,Z,original_author_list)

# print 'accuracy_native',logreg.score(native_test_vectors,test_labels)
# print 'accuracy_native',np.average(np.array(Z)==np.array(test_labels))
# print subprocess.check_output( 'python fscore.py GOLD.txt MY.txt' ,shell=True)

logfile.close()

# Saving the objects:
# with open('objs.pickle'+sys.argv[2], 'w') as f:
#     pickle.dump([vectorizer,logreg], f)