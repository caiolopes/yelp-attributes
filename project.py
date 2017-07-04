import sys
import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()


    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class StemmerTokenizer(object):
    def __init__(self):
        self.porter = PorterStemmer()


    def __call__(self, doc):
        return [self.porter.stem(t) for t in word_tokenize(doc)]


def show_most_informative_features(f, vectorizer, clf, n=25):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2), file=f)


def preprocess(s):
    s = s.strip().lower()
    s = ''.join(i for i in s if not i.isdigit())
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'(.)\1+', r'\1\1', s) 
    s = s.replace('_', '')

    for ch in string.punctuation:
        s = s.replace(ch, ' ')

    s = ' '.join(s.split())
    return s


if len(sys.argv) > 1:
    df = pd.read_csv('data/reviews.csv')
    df = df.astype('str').groupby('business_id').agg(lambda x: ''.join(set(x))).reset_index()

    train, test = train_test_split(df, test_size = 0.25)

    vect = CountVectorizer(preprocessor=preprocess, tokenizer=LemmaTokenizer(), stop_words='english')
    # vect = TfidfVectorizer(preprocessor=preprocess, use_idf=False, tokenizer=LemmaTokenizer(), stop_words='english')
    # vect = TfidfVectorizer(preprocessor=preprocess, use_idf=False, stop_words='english')
    # vect = TfidfVectorizer(preprocessor=preprocess, stop_words='english')
    # vect = TfidfVectorizer(preprocessor=preprocess, tokenizer=StemmerTokenizer(), stop_words='english')
    # vect = TfidfVectorizer(preprocessor=preprocess, tokenizer=LemmaTokenizer(), stop_words='english')

    print('Training size of', len(train), 'businesses')
    train_corpus = vect.fit_transform(train['text'].as_matrix())

    # Y = test.astype('str').groupby('business_id').agg(lambda x: ''.join(set(x))).reset_index()
    Y = test
    print('Testing on', len(Y), 'businesses')
    test_corpus = vect.transform(Y['text'].as_matrix())

    # text_clf = Pipeline([
        # ('vect', TfidfVectorizer(preprocessor=preprocess, stop_words='english')),
        # ('naive', MultinomialNB(alpha=0.03))
    # ])

    f = open(sys.argv[1] , 'w')
    for test_col in df.columns[4:]:
        # print(test_col)
        # score = cross_val_score(text_clf, df['text'], df[test_col], n_jobs=-1, cv=10)
        # print(score)
        # print(test_col, file=f)
        # print(score, file=f)

        print(test_col)
        print(test_col, file=f)
        clf = MultinomialNB(alpha=0.03).fit(train_corpus, train[test_col].as_matrix().astype('str'))

        expected = Y[test_col].as_matrix().astype('str')
        predicted = clf.predict(test_corpus)

        acc = np.mean(predicted == expected)

        print(acc)
        print(acc, file=f)
        print(metrics.classification_report(expected, predicted), file=f)
        show_most_informative_features(f, vect, clf)

else:
    print('usage: python', sys.argv[0], 'output_filename')

