from __future__ import division

import nltk
from nltk.probability import *

# a class for providing descriptive statistics about tokenized texts
class TextDescriptor:
	def __init__(self, text):
		self.text = text

	def lexical_richness(self):
		return len(self.text) / len(set(self.text))

	def frequencies(self):
		return nltk.FreqDist(self.text).items()

	def hapaxes(self):
		return nltk.FreqDist(self.text).hapaxes()