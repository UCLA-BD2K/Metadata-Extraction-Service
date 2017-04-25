from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from pymongo import MongoClient
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import TransformerMixin
from scipy import interp
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import os
from os import listdir
from os.path import isfile, join
import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, Normalizer
import json
from json import JSONDecoder
from functools import partial
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from matplotlib.font_manager import FontProperties
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import subprocess
import shutil
import time

# Still under lot of experimentation. Currently uses full text, uses tf-idf with Truncated SVD
# Hard voting classifier on NN, Logistic regression and SVM.

repo_list = ['github', 'github.com', 'sourceforge', 'sourceforge.net', 'bioconductor', 'bioconductor.org']

# List of parameters to test when running Grid Search to find best pair of values

svm_tuned_parameters = [
    {'svc__kernel': ['rbf', 'linear', 'sigmoid'], 'svc__gamma': [1e-3, 1e-4, 0.01, 0.1, 'auto', 1, 10],
     'svc__C': [1, 5, 10, 15, 20],
     'svc__class_weight': ['balanced', None]}]
#
sgd_tuned_parameters = [{'sgd__loss': ['hinge', 'log', 'modified_huber'],
                         'sgd__penalty': ['l2', 'elasticnet'],
                         'sgd__n_iter': [500, 1000, 2000, 5000],
                         'sgd__class_weight': ['balanced', None, {0: 1, 1: 10}],
                         'sgd__alpha': [1e-1, 1e-2, 1e-3]}]

# log_tuned_parameters = [{'log__penalty': ['l2'],
#                          'log__C': [0.01, 0.1, 1, 16, 64, 256, 1024, 4096],
#                          'log__solver': ['sag'],
#                          'log__max_iter': [200, 300, 500, 100],
#                          'log__class_weight': ['balanced', None, {"0": 20, "1": 1}, {"0": 15, "1": 1}]}]

mlp_tuned_parameters = [{'mlp__hidden_layer_sizes': [(1, 1), (2, 2), (3, 2), (4, 2)],
                         'mlp__activation': ['logistic', 'tanh', 'relu'],
                         'mlp__algorithm': ['l-bfgs'],
                         'mlp__alpha': [1e-3, 1e-4, 1e-2],
                         'mlp__max_iter': [300]}]

precison_list = []
recall_list = []
classifier_list = [SVC(gamma=0.1, kernel='rbf', C=10, class_weight='balanced'),
                   MLPClassifier(activation='logistic', alpha=0.001, hidden_layer_sizes=(4, 2),
                                 max_iter=1000),
                   SGDClassifier(n_jobs=-1, n_iter=500, penalty='elasticnet', alpha=0.001,
                                 loss='log', shuffle=True, class_weight='balanced'),
                   ]
path = None


# Returns list of abstracts

class AbstractTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        abstracts = []
        for obj in X:
            abstracts.append(obj['abstract'])
        return abstracts

    def fit(self, X, y=None, **fit_params):
        return self


# Returns list of titles

class TitleTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        titles = []
        for obj in X:
            titles.append(obj['title'])
        return titles

    def fit(self, X, y=None, **fit_params):
        return self


# Returns dict of features

class DictTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        result = []
        for obj in X:
            output = dict()
            output['has_repo'] = obj['has_repo']
            output['has_colon'] = obj['has_colon']
            output['tech_count'] = obj['tech_count']
            result.append(output)
        return result

    def fit(self, X, y=None, **fit_params):
        return self


class Scaler(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        return preprocessing.scale(X, with_mean=False)

    def fit(self, X, y=None, **fit_params):
        return self


# class Normalizer(BaseEstimator, TransformerMixin):
#     def transform(self, X, **transform_params):
#         return preprocessing.normalize(X, norm='l2')
#
#     def fit(self, X, y=None, **fit_params):
#         return self


pipeline = Pipeline([
    # Use FeatureUnion to combine the features from subject and body
    # ('union', FeatureUnion(
    #     transformer_list=[
    #         # ('abstract', Pipeline([
    #         #     ('extract', AbstractTransformer()),
    #         #     ('tf-idf', TfidfVectorizer()),
    #         # ])),
    #         # ('title', Pipeline([
    #         #     ('extract', TitleTransformer()),
    #         #     ('tf-idf', TfidfVectorizer()),
    #         # ])),
    #         # ('other', Pipeline([
    #         #     ('extract', DictTransformer()),
    #         #     ('dict', DictVectorizer()),
    #         # ])),
    #         ('text', Pipeline([
    #             ('extract', TfidfVectorizer(ngram_range=(2, 2))),
    #         ])),
    #
    #     ],
    # )),
    # ('scale', Pipeline([
    #     ('extract', Scaler())
    # ])),
    # ('truncate', Pipeline([
    #     ('svd', TruncatedSVD(n_components=100))
    # ])),
    # ('normalize', Pipeline([
    #     ('extract', Normalizer())
    # ])),
    # Use a SVC classifier on the combined features
    # ('kbest', SelectKBest(chi2, k=1000)),
    ('sgd', SGDClassifier(n_jobs=-1, n_iter=500, penalty='elasticnet', alpha=0.001,
                          loss='log', shuffle=True, class_weight='balanced'))
    # ('knn', KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine'))
    # ('mlp', MLPClassifier(activation='logistic', algorithm='l-bfgs', alpha=0.001, hidden_layer_sizes=(4, 2),
    #                       max_iter=1000))
    # ('svc', SVC(gamma=0.001, kernel='linear', C=1, class_weight=None))

])


def filter_text(text):
    text = ' '.join([word for word in text.split() if word not in set((stopwords.words('english')))])
    return text.lower()


# def fetch_from_mongo():
#     tool_count = 0
#     non_tool_count = 0
#     tot_tech_count = 0
#     tot_non_tech_count = 0
#     x = []
#     y = []
#     client = MongoClient('mongodb://BD2K:ucla4444@ds145415.mlab.com:45415/dois')
#     db = client['dois']
#     info = db['numbers']
#     for article in info.find():
#         output = dict()
#         title = filter_text(article['title'])
#         tract = filter_text(article['abstract'])
#         output['title'] = article['title']
#         output['abstract'] = article['abstract']
#         tech_count = 0
#         has_repo = False
#         for word in word_list:
#             if word in title or word in tract:
#                 tech_count += 1
#         for word in repo_list:
#             if word in title or word in tract:
#                 has_repo = True
#                 break
#         output['tech_count'] = tech_count
#         output['has_repo'] = has_repo
#         try:
#             if str(article['title'].encode('utf-8')).find(':') != -1:
#                 output['has_colon'] = True
#             else:
#                 output['has_colon'] = False
#         except Exception as e:
#             print e
#         if article['is_tool']:
#             output['is_tool'] = True
#             tool_count += 1
#             y.append(1)
#             tot_tech_count += output['tech_count']
#         else:
#             output['is_tool'] = False
#             non_tool_count += 1
#             tot_non_tech_count += output['tech_count']
#             y.append(0)
#
#         x.append(output)
#
#         # print "Average number of tech words in tools is " + str(tot_tech_count/tool_count)
#         # print "Average number of tech words in non tools is " + str(tot_non_tech_count/non_tool_count)
#     return x, y


def plot_roc(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_curve():
    N = 3
    ind = np.arange(N)
    width = 0.4
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, precison_list, width, color='r')
    rects2 = ax.bar(ind + width, recall_list, width, color='y')
    ax.set_ylabel('Presicion and recall scores')
    ax.set_title('Classifier performance')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('Support Vector Machine', 'Neural Network', 'Logistic Regression'))
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend((rects1[0], rects2[0]), ('Precision', 'Recall'), bbox_to_anchor=(1.1, 1.05))

    plt.show()


def fetch_data():
    tool_path = "no_ref_tools/"
    non_tool_path = "no_ref_non_tools/"
    y = []
    try:
        tools = [tool_path + f for f in listdir(tool_path) if isfile(
            join(tool_path, f))]
        y += [1] * len(tools)
        non_tools = [non_tool_path + f for f in listdir(non_tool_path) if isfile(
            join(non_tool_path, f))]
        y += [0] * len(non_tools)
    except Exception as e:
        print e
        print "could not retrieve data, aborting..."
        sys.exit(1)
    return tools + non_tools, y


def extract_text(input_files):
    x = []
    for f in input_files:
        with open(f, "r") as input_file:
            x.append(input_file.read())
    return x


def normalize_scale(X_train, X_test):
    vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_df=0.25, min_df=2, use_idf=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    svd = TruncatedSVD(100, algorithm='arpack')
    lsa = make_pipeline(svd, Normalizer(copy=False))
    X_train = lsa.fit_transform(X_train)
    X_test = lsa.transform(X_test)
    return X_train, X_test


def store_data(x, y):
    with open(path+"/util/data.json", "w") as output:
        for doc, category in zip(x, y):
            dictionary = dict()
            dictionary['class'] = category
            dictionary['text'] = doc
            json.dump(dictionary, output)


def fetch_from_file():
    x = []
    y = []
    with open(path+"/util/data.json", "r") as input_file:
        for obj in json_parse(input_file):
            x.append(obj['text'])
            y.append(obj['class'])
    return x, y


def json_parse(fileobj, decoder=JSONDecoder(), buffersize=2048):
    buf = ''
    for chunk in iter(partial(fileobj.read, buffersize), ''):
        buf += chunk
        while buf:
            try:
                result, index = decoder.raw_decode(buf)
                yield result
                buf = buf[index:]
            except ValueError:
                # Not enough data to decode, read more
                break


def get_test_set(directory):
    files = os.listdir(directory)
    text_dir = 'textDir/'
    if os.path.isdir(text_dir):
        shutil.rmtree(text_dir)
    os.makedirs(text_dir)
    test_files = []
    for file in files:
        out_filename = text_dir + file.replace('.pdf', '.txt')
        subprocess.call(['pdftotext', '-enc', 'UTF-8', directory + file, out_filename])
        test_files.append(file)
    x_test = []
    for file in os.listdir(text_dir):
        with open(text_dir+file, "r") as text_file:
            x_test.append(text_file.read())
    shutil.rmtree(text_dir)
    return x_test, test_files


def main(directory, tools_directory, non_tools_dir):
    global path
    path = sys.path[0]
    start = time.time()
    if directory is None or not os.path.isdir(directory):
        print "Please input directory containing pdf publications to classify"
        sys.exit(1)
    x_train, y_train = fetch_from_file()
    x_test, test_files = get_test_set(directory)
    # Just for testing, update machine learning part later

    x_train, x_test = normalize_scale(x_train, x_test)
    classifier = VotingClassifier([('first', classifier_list[0]), ('second', classifier_list[1]),
                                   ('second', classifier_list[2])])
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    if os.path.isdir(tools_directory):
        shutil.rmtree(tools_directory)
    os.makedirs(tools_directory)

    if os.path.isdir(non_tools_dir):
        shutil.rmtree(non_tools_dir)
    os.makedirs(non_tools_dir)

    for num, pub in zip(y_pred, test_files):
        if num:
            shutil.copy2(directory + pub, tools_directory + pub)
        else:
            shutil.copy2(directory + pub, non_tools_dir + pub)

    print "Classification:    Seconds taken: " + str(time.time() - start)

    # input_files, y = fetch_data()
    # x = extract_text(input_files)
    # x = [filter_text(doc) for doc in x]
    # store_data(x,y)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2)
    # X_train, X_test = normalize_scale(X_train, X_test)
    # for i in range(len(classifier_list)):
    #     classifier_list[i].fit(X_train, y_train)
    #     y_pred = classifier_list[i].predict(X_test)
    #     print classifier_list[i].score(X_test, y_test)
    #     print classification_report(y_test, y_pred)
    #     print confusion_matrix(y_test, y_pred)
    #     precison_list.append(precision_score(y_test, y_pred))
    #     recall_list.append(recall_score(y_test, y_pred))
    #     print y_test
    #     print y_pred
    # plot_curve()
    # clf = GridSearchCV(pipeline, param_grid=sgd_tuned_parameters, cv=5, verbose=5)
    # clf.fit(X_train, y_train)
    # print "Best parameters set found on development set:"
    # print '\n'
    # print clf.best_params_
    # print'\n'
    # print "Grid scores on development set:"
    # print '\n'
    # for params, mean_score, scores in clf.grid_scores_:
    #     print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
    # print '\n'
    #
    # print("Detailed classification report:")
    # print '\n'
    # print "The model is trained on the full development set."
    # print "The scores are computed on the full evaluation set."
    # print '\n'
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print classification_report(y_true, y_pred)
    # print confusion_matrix(y_true, y_pred)
    # print '\n'

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
