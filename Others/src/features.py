import numpy as np
import sklearn
import nltk
import nltk.data
import string

#returns the string corresponding to the document specified by doc_path
def doc_to_string(doc_path):
	doc_string="";
	doc_file = open(doc_path, "r");
	curr_line = doc_file.readline();
	while(curr_line):
		doc_string = doc_string + curr_line[0:len(curr_line)-1]+" ";
		curr_line = doc_file.readline();
	return doc_string;

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

#returns the number of distinct tokens in text
def count_uniq_tokens(text):
	tokens = nltk.word_tokenize(text);
	num_uniq_tokens = len(set(tokens));
	return num_uniq_tokens;

def insert_pos_tags(text):
	new_text = "";
	tokens_tags = nltk.pos_tag(nltk.word_tokenize(text));
	for pair in tokens_tags:
		new_text = new_text + pair[0] + " " + pair[1] + " ";
	return new_text;

#returns the array with additional feature values
def extra_feats(text):
	feat_array=[];
	feat_array.append(count_tokens(text));			#number of tokens
	feat_array.append(count_uniq_tokens(text));		#number of unique tokens
	feat_array.append(count_sents(text));			#number of sentences
	feat_array.append(frac_puncs(text));			#fraction of punctuations
	feat_array.append(avg_token_len(text));			#average token length
	feat_array.append(stdev_token_len(text));		#standard deviation of token lengths
	feat_array.append(avg_sent_len1(text));			#average sentence length (wrt words)
	feat_array.append(avg_sent_len2(text));			#average sentence length (wrt chars)
	feat_array.append(stdev_sent_len1(text));		#standard deviation of sentence length (wrt words)
	feat_array.append(stdev_sent_len2(text));		#standard deviation of sentence length (wrt chars)
	feat_array.append(num_digits(text));			#number of digits	return feat_array;
	return feat_array;

doc_str = doc_to_string("doc.txt");
#arr = extra_feats(doc_str);
doc_str2 = insert_pos_tags(doc_str);
print doc_str2;

#n = count_sents("My name is Abhishek. I study at I.I.T. Delhi.");
#n = count_tokens(doc_str);
#n = stdev_token_len(doc_str);
#print n;

#num of chars
#text = nltk.Text("My name is Abhishek. I study at I.I.T. Delhi.");
#n = len(text);

#n = len(nltk.word_tokenize("My name is Abhishek. I study at I.I.T. Delhi."));
"""
str = "My name is Abhishek. I study at IIT Delhi."
words1 = nltk.word_tokenize(str);
words2 = nltk.word_tokenize(str.translate(None, string.punctuation));
print words1;
print words2;
"""
