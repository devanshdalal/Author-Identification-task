import nltk
import nltk.data

#some thin wrappers around the core NLTK tokenizers

class SentenceTokenizer:
	def __init__(self):
		self.trainer = nltk.data.load('tokenizers/punkt/english.pickle')

	def make_sentence_tokens(self, text):
		# some simple cleaning
		clean_text = text.replace('\n','').replace('\t','').replace('  ',' ')
		return self.trainer.tokenize(clean_text)

class WordTokenizer:
	def __init__(self):
		pass

	def make_word_tokens(self, text):
		clean_text = text.replace('\n','').replace('\t','').replace('  ',' ')
		return nltk.word_tokenize(clean_text)