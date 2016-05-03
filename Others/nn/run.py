#! /usr/local/bin/python

# runs extract.py and compiles and runs nn-nlp.c

#  Anthony Pasqualoni
#  Independent Study: Neural Networks and Pattern Recognition
#  Adviser: Dr. Hrvoje Podnar, SCSU
#  June 27, 2006

import os
import random
import sys

# amount of runs:
if (len(sys.argv) > 1):
   runs = sys.argv[1]
else:
   runs = 1

runs = int(runs)
random.seed(1)
arg = [0,0,0,0,0,0,0,0]

for i in range(runs):

   # random parameters for threshold values in extract.py:
   # (used only if use_rand > 0 in extract.py)
   arg[0] = (random.uniform(1.2,8.0))
   arg[1] = (random.uniform(0.2,0.7))
   arg[2] = (random.uniform(3.0,8.0))
   arg[3] = (random.uniform(0.2,0.7))
   arg[4] = (random.uniform(4.0,10.0))
   arg[5] = (random.uniform(0.2,0.7))
   arg[6] = (random.uniform(4.0,10.0))
   arg[7] = (random.uniform(0.2,0.7))

   # print random parameters to stdout:
   if (len(sys.argv) > 1):
      out = "parameters: \n"
      for j in range(len(arg)):
         out += str(arg[j])
         if not (j%2):
            out += " "
         else:
            out += "\n"
      os.system("echo -n '" + out + "'")

   # run feature extraction script:
   cmd = "python extract.py " 
   for j in range(len(arg)):
       cmd += str(arg[j]) + " "
   cmd += " > extract.out"

   print cmd
   os.system(cmd)
   # exit(0)

   # compile and run neural network:
   os.system("gcc nn-nlp.c -lm")
   os.system("./a.out")

