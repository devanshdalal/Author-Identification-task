#! /usr/local/bin/python

#  This script extracts the following features from ASCII text files:
#    single word count
#    horizontal word pair count
#    vertical word pair count
#    letter pair count

#  Anthony Pasqualoni
#  Independent Study: Neural Networks and Pattern Recognition
#  Adviser: Dr. Hrvoje Podnar, SCSU
#  June 27, 2006

import random
import re
import sys

def report_word_freq (text_a,text_b,high_thresh,low_thresh,sonnet_amount):
# returns lists of high and low frequency words based on threshold values
   
   count_a = {}
   count_b = {}

   high = []
   low = []

   # create dictionary of word counts for each text:
   for s in range(sonnet_amount):
      sonnet = get_sonnet(text_a,s)
      for word in sonnet.split():
         if word in count_a:
            count_a[word] += 1
         else:
            count_a[word] = 1
            
   for s in range(sonnet_amount):
      sonnet = get_sonnet(text_b,s)
      for word in sonnet.split():
         if word in count_b:
            count_b[word] += 1
         else:
            count_b[word] = 1

   # print results and determine difference in frequency between the two
   # texts (contrast):
   for word in count_a:
      print "%18s %3i" % (word, count_a[word],),
      if word in count_b:
         contrast =  float(count_a[word]) / float(count_b[word])
         print "%3i    %3.5f" % (count_b[word], contrast)
      else:
         if (count_a[word]) >= 2:
            contrast = float(count_a[word])
         else:
            contrast = 1
         print "*0     %3.5f" % (contrast)
      if ( contrast > high_thresh ):
         high.append(word)
      if ( contrast < low_thresh ):
         low.append(word)

   for word in count_b:
      if count_b[word] >= 4 and word not in count_a:
         print "* not in shakespeare: ", word,
         print count_b[word]
         contrast = count_b[word]
         low.append(word)


   return (high,low)

def report_letter_pair_count(text_a,text_b,high_threshold,low_threshold,sonnet_amount):
# returns lists of high and low frequency letter pairs based on threshold values

   pair_dict_a = {}
   pair_dict_b = {}

   high = []
   low = []

   # create dictionary of word counts for each text:
   for s in range(sonnet_amount):
      sonnet = get_sonnet(text_a,s)
      for i in range( len(sonnet) - 1):
         a = sonnet[i]
         b = sonnet[i+1]
         if a.isalpha() and b.isalpha():
            pair = a + b
            if pair not in pair_dict_a:
               pair_dict_a[pair] = 1
            else:
               pair_dict_a[pair] += 1
               
   for s in range(sonnet_amount):
      sonnet = get_sonnet(text_b,s)
      for i in range( len(sonnet) - 1):
         a = sonnet[i]
         b = sonnet[i+1]
         if a.isalpha() and b.isalpha():
            pair = a + b
            if pair not in pair_dict_b:
               pair_dict_b[pair] = 1
            else:
               pair_dict_b[pair] += 1

   # print results and determine difference in frequency between the two
   # texts (contrast):
   for string in pair_dict_a:
      print "%10s %3i" % (string,pair_dict_a[string],),
      if string in pair_dict_b:
         contrast =  float(pair_dict_a[string]) / float(pair_dict_b[string])
         print "%3i    %3.5f" % (pair_dict_b[string], contrast)
      else:
         if pair_dict_a[string] >= 4:
            contrast =  float(pair_dict_a[string])
         else:
            contrast = 1
         print "*0     %3.5f" % (contrast)
      if ( contrast > high_threshold ):
         high.append(string)
      if ( contrast < low_threshold ):
         low.append(string)

   for string in pair_dict_b:
      if pair_dict_b[string] >= 4 and string not in pair_dict_a:
         print "* not in shakespeare: ", string,
         print pair_dict_b[string]
         contrast = pair_dict_b[string]
         low.append(string)

   return high,low

def report_word_pair_count(text_a,text_b,high_threshold,low_threshold,sonnet_amount):
# returns lists of high and low frequency adjacent word pairs based on threshold values

   pair_dict_a = {}
   pair_dict_b = {}

   high = []
   low = []

   # create dictionary of word counts for each text:
   for s in range(sonnet_amount):
      sonnet_words = get_sonnet(text_a,s).split()
      for i in range(len(sonnet_words) - 1):
         a = sonnet_words[i]
         b = sonnet_words[i+1]
         pair = a + " " + b
         if pair not in pair_dict_a:
            pair_dict_a[pair] = 1
         else:
            pair_dict_a[pair] += 1
               
   for s in range(sonnet_amount):
      sonnet_words = get_sonnet(text_b,s).split()
      for i in range(len(sonnet_words) - 1):
         a = sonnet_words[i]
         b = sonnet_words[i+1]
         pair = a + " " + b
         if pair not in pair_dict_b:
            pair_dict_b[pair] = 1
         else:
            pair_dict_b[pair] += 1

   # print results and determine difference in frequency between the two
   # texts (contrast):
   for string in pair_dict_a:
      if pair_dict_a[string] >= 3:
         print "%36s %3i" % (string,pair_dict_a[string],),
         if string in pair_dict_b:
            contrast =  float(pair_dict_a[string]) / float(pair_dict_b[string])
            print "%3i    %3.5f" % (pair_dict_b[string], contrast)
         else:
            if pair_dict_a[string] >= 4:
               contrast = float(pair_dict_a[string])
            else:
               contrast = 1
            print "       %3.5f" % (contrast)
         if ( contrast > high_threshold ):
            high.append(string)
         if ( contrast < low_threshold ):
            low.append(string)

   for string in pair_dict_b:
      if pair_dict_b[string] >= 4 and string not in pair_dict_a:
         print "not in shakespeare: ", string,
         print pair_dict_b[string]
         contrast = pair_dict_b[string]
         low.append(string)

   return high,low

def report_vertical_word_pair_count(text_a,text_b,high_threshold,low_threshold,sonnet_amount):
# returns lists of high and low frequency vertical word pairs based on threshold values
   
   pair_dict_a = {}
   pair_dict_b = {}

   high = []
   low = []

   # create dictionary of word counts for each text:
   sonnet_lines = text_a.splitlines(0)
   for i in range(len(sonnet_lines) - 1):
      line_a = sonnet_lines[i]
      words_a = line_a.split()
      line_b = sonnet_lines[i+1]
      words_b = line_b.split()
      for i in range(len(words_a)):
         if i < len(words_b):
            pair = words_a[i] + " " + words_b[i]
            if pair not in pair_dict_a:
               pair_dict_a[pair] = 1
            else:
               pair_dict_a[pair] += 1

   sonnet_lines = text_b.splitlines(0)
   for i in range(len(sonnet_lines) - 1):
      line_a = sonnet_lines[i]
      words_a = line_a.split()
      line_b = sonnet_lines[i+1]
      words_b = line_b.split()
      for i in range(len(words_a)):
         if i < len(words_b):
            pair = words_a[i] + " " + words_b[i]
            if pair not in pair_dict_b:
               pair_dict_b[pair] = 1
            else:
               pair_dict_b[pair] += 1

   # print results and determine difference in frequency between the two
   # texts (contrast):
   for string in pair_dict_a:
      if pair_dict_a[string] >= 3:
         print "%36s %3i" % (string,pair_dict_a[string],),
         if string in pair_dict_b:
            contrast =  float(pair_dict_a[string]) / float(pair_dict_b[string])
            print "%3i    %3.5f" % (pair_dict_b[string], contrast)
         else:
            if pair_dict_a[string] >= 4:
               contrast = float(pair_dict_a[string])
            else:
               contrast = 1
            print "       %3.5f" % (contrast)
         if ( contrast > high_threshold ):
            high.append(string)
         if ( contrast < low_threshold ):
            low.append(string)

   for string in pair_dict_b:
      if pair_dict_b[string] >= 4 and string not in pair_dict_a:
         print "not in shakespeare: ", string,
         print pair_dict_b[string]
         contrast = pair_dict_b[string]
         low.append(string)

   return high, low
   

def get_sonnet(text,id):
# returns (id)th sonnet in text (text is formatted with 3 lines between each sonnet)

   first_line = id * 17 + 3
   last_line = first_line + 13

   sonnet = ""
   count = 0
   for line in text.splitlines(1):
      if count >= first_line and count <= last_line:
         sonnet += line
      count += 1

   return sonnet
         
def sonnet_word_count (text,string_list):
# returns amount of words in text that are included in string_list
   
   count = 0
   for word in text.split():
      if word in string_list:
         count += 1

   return count


def sonnet_string_count (text,string_list):
# returns amount of strings (e.g. letter pairs) in text that are included in string_list

   count = 0
   for pair in string_list:
      count += text.count(pair)

   return count


def save_sonnet_data(sonnet_amount,high_words,low_words,high_pairs,low_pairs,high_word_pairs,low_word_pairs,high_vert,low_vert):
# call feature extraction functions for each text using high and low frequency lists.
# save results to file 'features.txt'.
# letter pair counts are not used since they tend to reduce pattern recognition accuracy

   f = open ('features.txt','w')

   for i in range(sonnet_amount):
   
      # extract features from shakespeare sonnet:
      sonnet = get_sonnet(will,i)
      high_word_count = sonnet_word_count(sonnet,high_words)
      low_word_count  = sonnet_word_count(sonnet,low_words)
      high_word_pair_count = sonnet_string_count(sonnet,high_word_pairs)
      low_word_pair_count  = sonnet_string_count(sonnet,low_word_pairs)
      high_word_vert_count = sonnet_string_count(sonnet,high_vert)
      low_word_vert_count  = sonnet_string_count(sonnet,low_vert)

      line_a = "%3i,%3i,%3i,%3i,%3i,%3i,  1, 1000\n" % (high_word_count,low_word_count,high_word_pair_count,low_word_pair_count,high_word_vert_count,low_word_vert_count)
   
      # extract features from miscellaneous sonnet:
      sonnet = get_sonnet(misc,i)
      high_word_count = sonnet_word_count(sonnet,high_words)
      low_word_count  = sonnet_word_count(sonnet,low_words)
      high_word_pair_count = sonnet_string_count(sonnet,high_word_pairs)
      low_word_pair_count  = sonnet_string_count(sonnet,low_word_pairs)
      high_word_vert_count = sonnet_string_count(sonnet,high_vert)
      low_word_vert_count  = sonnet_string_count(sonnet,low_vert)

      line_b = "%3i,%3i,%3i,%3i,%3i,%3i, -1, 1000\n" % (high_word_count,low_word_count,high_word_pair_count,low_word_pair_count,high_word_vert_count,low_word_vert_count)

      f.write (line_a)
      f.write (line_b)

def save_graph_data(high_words,low_words,high_pairs,low_pairs,high_word_pairs,low_word_pairs,high_vert,low_vert):
# save data for producing graphs with gnuplot; not required for neural network

   f_will = open ('shakespeare.dat','w')
   f_misc = open ('spenser-smith-griffin.dat','w')
   #f_misc = open ('./spenser-smith-griffin-drayton-full.dat','w')

   sonnet_amount = sonnets_per_text/2

   for i in range(sonnet_amount):
   
      # extract features from shakespeare sonnet:
      sonnet = get_sonnet(will,i)
      high_word_count = sonnet_word_count(sonnet,high_words)
      low_word_count  = sonnet_word_count(sonnet,low_words)
      high_word_pair_count = sonnet_string_count(sonnet,high_word_pairs)
      low_word_pair_count  = sonnet_string_count(sonnet,low_word_pairs)
      high_word_vert_count = sonnet_string_count(sonnet,high_vert)
      low_word_vert_count  = sonnet_string_count(sonnet,low_vert)

      line = "%i   %i\n" % (high_word_count + high_word_vert_count + high_word_pair_count,
                            low_word_count + low_word_vert_count + low_word_pair_count) 

      f_will.write (line)

      # extract features from sonnets by other authors:
      sonnet = get_sonnet(misc,i)
      high_word_count = sonnet_word_count(sonnet,high_words)
      low_word_count  = sonnet_word_count(sonnet,low_words)
      high_word_pair_count = sonnet_string_count(sonnet,high_word_pairs)
      low_word_pair_count  = sonnet_string_count(sonnet,low_word_pairs)
      high_word_vert_count = sonnet_string_count(sonnet,high_vert)
      low_word_vert_count  = sonnet_string_count(sonnet,low_vert)

      line = "%i   %i\n" % (high_word_count + high_word_vert_count + high_word_pair_count,
                            low_word_count + low_word_vert_count + low_word_pair_count) 
      f_misc.write (line)

def extract_and_save():
# determine threshold either randomly (if use_rand > 0) or using
# constants. Threshold is the minimum/maximum difference between
# word/letter pair counts of the two texts being compared.

   use_rand = 0

   high_thresh_a = 3.46820317312 
   low_thresh_a = 0.367356090223
   if (use_rand == 2):
      high_thresh_a = float(sys.argv[1])
      low_thresh_a =  float(sys.argv[2])
   high_w,low_w = report_word_freq(will, misc, high_thresh_a, low_thresh_a, sonnets_per_text/2)
   print "high words: ", high_w
   print "low words: ", low_w
   
   high_thresh_b = 3.38310197037 
   low_thresh_b = 0.366919825994
   if (use_rand == 2):
      high_thresh_b = float(sys.argv[3])
      low_thresh_b =  float(sys.argv[4])
   high_p, low_p = report_letter_pair_count(will,misc, high_thresh_b, low_thresh_b, sonnets_per_text/2)
   print "high letter pairs: ", high_p
   print "low letter pairs: ", low_p

   high_thresh_c = 5.04054893116 
   low_thresh_c = 0.445208715939
   if (use_rand >= 1):
      high_thresh_c = float(sys.argv[5])
      low_thresh_c =  float(sys.argv[6])
   high_word_p, low_word_p = report_word_pair_count(will,misc,high_thresh_c,low_thresh_c,sonnets_per_text/2)
   print "high word pairs: ",high_word_p
   print "low word pairs: ",low_word_p

   high_thresh_d = 4.16058982088;
   low_thresh_d = 0.23356840741;
   if (use_rand >= 1):
      high_thresh_d = float(sys.argv[7])
      low_thresh_d =  float(sys.argv[8])
   high_vert_p, low_vert_p = report_vertical_word_pair_count(will,misc,high_thresh_d,low_thresh_d,sonnets_per_text/2)
   print "high vertical pairs: ",high_vert_p
   print "low vertical pairs: ",low_vert_p

   print "thresholds: ",
   print high_thresh_a, low_thresh_a, high_thresh_b, low_thresh_b, high_thresh_c, low_thresh_c, high_thresh_d, low_thresh_d

   #save_graph_data(high_w,low_w,high_p,low_p,high_word_p,low_word_p,high_vert_p,low_vert_p)
   
   save_sonnet_data(sonnets_per_text,high_w,low_w,high_p,low_p,high_word_p,low_word_p,high_vert_p,low_vert_p)

# ====================================== end of function definitions =======================================

# define regular expression for non-alphanumeric characters that occur in texts:
nonalpha = re.compile('\.|\,|\:|\;|\?|\!|\'|\-|\_|\(|\)')

sonnet_amt = 200   # amount of sonnets in shakespeare.txt plus text of Spenser et al.
sonnets_per_text = sonnet_amt/2
   
max_lines = 17 * sonnets_per_text
max_line_width = 50

# read text files:
f = open ('shakespeare.txt','r')
will_orig = f.read()

# f = open ('spenser-smith-griffin.txt','r') 
f = open ('spenser-smith-griffin-drayton.txt','r') 
misc_orig = f.read()

# use only specified amount of lines and strip leading white space:
will = ""
count = 0
for line in will_orig.splitlines(1):
   if count < max_lines:
      count += 1
      if not line.isspace():
         will += line.lstrip()
      else:
         will += line
               
misc = ""
count = 0
for line in misc_orig.splitlines(1):
   if count < max_lines:
      count += 1
      if not line.isspace():
         misc += line.lstrip()
      else:
         misc += line

# convert to lower case and delete non-alphanumeric characters:
will = will.lower()
will = nonalpha.sub("",will)

misc = misc.lower()
misc = nonalpha.sub("",misc)

#print will
#print misc

# call main function for feature extraction and for saving results to text file for
# neural network input:
extract_and_save()
