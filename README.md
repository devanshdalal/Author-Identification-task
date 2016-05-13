#Author-Identification-from-text

![Logo](https://cloud.githubusercontent.com/assets/5080310/13220998/186aac70-d99f-11e5-9527-6a8c97793f69.png)
---------------

## Problem
----------------
Authorship identification has been a very important and practical problem in Natural Language Processing. The problem is to identify the author of a document from a given list of possible authors. A large amount of work exists on this problem in literature. We develop ideas based on this work in order to build our own model for authorship identification. We also take a model from this work as a baseline for comparing the results. Our model for the task is a text classifier based on logistic regression which includes n-grams, style markers and document finger-printing as features. 


## Dataset
--------------
Reuter_50_50 is the dataset used. It is present in directories training/ , testing/ and all. It contains 50 text file for 50 authors. Each text file contains several lines for that author.

## Requirements
--------------
python with common ML and NLP libraries like Scikit-learn,Theano,Nltk etc.

## Organization
--------------
learner.py is the main file . run it to see the output.

<!-- ## History

TODO: Write history -->




## Credits
----------------
[Devansh Dalal](https://github.com/devanshdalal) <br>
[Abhishek]() <br>


## Contributing
----------------
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D
