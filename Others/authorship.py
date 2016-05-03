import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
 
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
import pandas as pd
import pylab as pl
from sklearn.cross_validation import train_test_split

 

"""
  open the file, remove mentions & links,  split it in  chunks of n words and return the n-word block
"""
def split_word_batches(filename, sizeOfChuck=100):
  #print "filename:", filename
  with open(filename, "r") as f: 
    lines = f.readlines() 
    text = "" 
    for l in lines:
      tokens = l.strip().split()
      if len(tokens)>1:
        text+= " " + tokens[1]

 	text = text.decode('utf-8').lower().split() 
  text = [ word for word in text if  word[0]!="@"  and not 'http' in word[0:4] and word[0]!='#' ]  
  start  = 0
  end = start+sizeOfChuck
  while True:
    start  = end
    end = start+sizeOfChuck 
    #print start, end 
    if end >=len(text):
      break
    #if start>=len(text):
    #	break 
    yield ' '.join(text[start:end])


 

def load_corpus(input_dir):
  from os import listdir
  from os.path import isfile, join
  trainfiles= [  f for f in listdir( input_dir ) if isfile(join(input_dir ,f)) ]

  trainset = []
  for f in trainfiles:

    label = f 
    df =  pd.read_csv( input_dir + "/" + f  , sep="\t", dtype={ 'id':object, 'text':object } )

    for row in df['text']:
      if type(row) is str:
        trainset.append(   { "label":label, "text": row }   )
      

  return trainset



"""
  train the models, using 10-fold-cv and LibLinear classification
"""
def train_model(trainset):


  #create two blocks of features, word anc character ngrams, size of 2
  #we can also append here multiple other features in general
  word_vector = TfidfVectorizer( analyzer="word" , ngram_range=(2,2), binary = False, max_features= 2000 )
  char_vector = TfidfVectorizer(ngram_range=(2, 3), analyzer="char", binary=False, min_df=0 , max_features=2000 )

  #ur vectors are the feature union of word/char ngrams
  vectorizer = FeatureUnion([  ("chars", char_vector),("words", word_vector)  ] )


  #corpus is a list with the n-word chunks
  corpus = []
  #classes is the labels of each chunk
  classes = []

  	#load training sets, for males & females

  	

  for item in trainset:    
  	corpus.append( item['text']  )
  	classes.append( item['label'] )


  print "num of training instances: ", len(classes)    
  print "num of training classes: ", len(set(classes))

    



  #fit the model of tfidf vectors for the coprus
  matrix = vectorizer.fit_transform(corpus)
 

  print "num of features: " , len(vectorizer.get_feature_names())

  print "training model"
  X =matrix.toarray()
  y = np.asarray(classes)
  
  model  = LinearSVC( loss='l1', dual=True)

  #scores = cross_validation.cross_val_score(  estimator = model,
	#		X = matrix.toarray(), 
  #    		y= np.asarray(classes), cv=10  )

  #http://scikit-learn.org/stable/auto_examples/plot_confusion_matrix.html
  #print scores



  #print "10-fold cross validation results:", "mean score = ", scores.mean(), "std=", scores.std(), ", num folds =", len(scores)

  #model.fit( X= matrix.toarray(), y= np.asarray(classes) )
  
  #predicted = model.predict(matrix.toarray())
  #cm = confusion_matrix(classes, predicted)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  y_pred = model.fit(X_train, y_train).predict(X_test)
  cm = confusion_matrix(y_test, y_pred)

  print(cm)

  pl.matshow(cm)
  pl.title('Confusion matrix')
  pl.colorbar()
  pl.ylabel('True label')
  pl.xlabel('Predicted label')
  pl.show()

  #below we are examples of several other learning algoritmhs we can use with skikit-learn	 
  #model  = SVC(kernel='sigmoid')
  #model = RandomForestClassifier()
  #model = KNeighborsClassifier(algorithm = 'kd_tree')
  #model  = LogisticRegression() #) #(kernel='sigmoid') #LogisticRegression() # 
  #model = NuSVC()
 
 


		 

if __name__=='__main__':  


	corpus = load_corpus("./corpus") 
 	print len(corpus), corpus[0:10]
	train_model( corpus )
	 