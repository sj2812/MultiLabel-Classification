import csv
import pandas as pd
import sys
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
# ignore all future warnings
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import multilabel_confusion_matrix
import re


maxInt=(sys.maxsize)
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


def read_data(s):
    dataset = []
    input_file = csv.reader(open(s,encoding='utf-8'))
    for row in input_file:
        if(len(row)>0):
            dataset.append(row)
    return pd.DataFrame(dataset,index=None)

content=read_data("content.csv")
label=read_data("label.csv")

labels=[]
for row in label[1]:
   labels.append(re.findall(r"'(.*?)'", row, re.DOTALL))

contentsub=content.head(19576)
labelsub=labels[0:19576]
x=contentsub[1]
y=labelsub
mlb=MultiLabelBinarizer()
Y=pd.DataFrame(mlb.fit_transform(y))
y_enc=mlb.fit_transform(y)
print(Y)
train_x, test_x, train_y, test_y = train_test_split(x, y_enc, test_size=0.35,shuffle=True)
simplefilter(action='ignore', category=FutureWarning)
classifier = Pipeline([
    ('vectorizer', CountVectorizer(lowercase=True,
                                   stop_words='english',
                                   max_df=0.8,
                                   min_df=10)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(train_x, train_y)
predicted_test = classifier.predict(test_x)

my_metrics = classification_report(test_y, predicted_test)
#print(my_metrics)
print("precision score")
print(metrics.precision_score(test_y, predicted_test,average='macro'))
print("Recall score")
print(metrics.recall_score (test_y, predicted_test,average='micro'))
print("Jaccard")
print(jaccard_score(test_y, predicted_test,average='macro'))
mcm = multilabel_confusion_matrix(test_y, predicted_test)
print(mcm)
print("Accuracy = ",accuracy_score(test_y,predicted_test))

#classifier_new.fit(x_train, y_train)
# predict
#predictions_new = classifier_new.predict(x_test)
# accuracy
#print("Accuracy = ",accuracy_score(test_y,predictions_new))
#print("\n")
