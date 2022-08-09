import csv
import sys
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from skmultilearn.problem_transform import ClassifierChain,LabelPowerset, BinaryRelevance
from sklearn.metrics import jaccard_score

import time

maxInt = sys.maxsize
labels=[]
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

#convert the preprocessed csv data to dataframe
def read_data(s):
    dataset=[]
    inputfile=csv.reader(open(s,encoding="utf-8"))
    for row in inputfile:
        if(len(row)>0):
            dataset.append(row)
    return pd.DataFrame(dataset)
content=read_data("content.csv")
label=read_data("label.csv")



labels=[]
for row in label[1]:
    labels.append(re.findall(r"'(.*?)'",row,re.DOTALL))

contentsub=content.head(10000)

x=content[1]
y=labels
mlb=MultiLabelBinarizer()
Y=pd.DataFrame(mlb.fit_transform(y))
y_enc=mlb.fit_transform(y) #binarise target variable
print(Y.head(10))
train_x,test_x,train_y,test_y=train_test_split(x,y_enc,test_size=0.35,shuffle=True) #holdout method to split train and test method
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,2), norm='l2') #tfidf vectoriser
#print(vectorizer.fit(["an ,apple, a ,day, keeps, the, doctor, away"]).vocabulary_)
vectorizer.fit(train_x)
vectorizer.fit(test_x)
x_train = vectorizer.transform(train_x)
y_train =train_y
x_test = vectorizer.transform(test_x)
y_test = test_y

#implement labelpowerset with random forest
classifier = LabelPowerset(
    classifier = RandomForestClassifier(n_estimators=100),
    require_dense = [False, True]
)
classifier.fit(x_train, y_train)

y_hat=classifier.predict(x_test)
lab_f1=metrics.f1_score(y_test, y_hat, average='micro')
lab_f2=metrics.f1_score(y_test, y_hat, average='macro')
lab_hamm=metrics.hamming_loss(y_test,y_hat)
print('Label Powerset micro F1-score:',round(lab_f1,3))
print('Label Powerset  macro F1-score:',round(lab_f2,3))
print('Label Powerset  Hamming Loss:',round(lab_hamm,3))
print("Exact matching score")
print(accuracy_score(y_test,y_hat))
print("precision score micro")
print(metrics.precision_score(y_test,y_hat,average='micro'))
print("precision score macro")
print(metrics.precision_score(y_test,y_hat,average='macro'))
print("Recall score micro")
print(metrics.recall_score (y_test,y_hat,average='micro'))
print("Recall score")
print(metrics.recall_score (y_test,y_hat,average='macro'))
print("Jaccard micro")
print(jaccard_score(y_test,y_hat,average='micro'))
print("Jaccard macro")
print(jaccard_score(y_test,y_hat,average='macro'))

#MLkNN implementation

classifier=MLkNN(k=3)

classifier.fit(x_train, y_train)


y_hat=classifier.predict(x_test)
lp_f1=metrics.f1_score(y_test, y_hat, average='micro')
lp_f2=metrics.f1_score(y_test, y_hat, average='macro')
lp_hamm=metrics.hamming_loss(y_test,y_hat)
print('MLKNN micro F1-score:',round(lp_f1,3))
print('MLKNN macro F1-score:',round(lp_f2,3))
print('MLKNN Hamming Loss:',round(lp_hamm,3))
print("Exact matching score")
print(accuracy_score(y_test,y_hat))
print("precision score micro")
print(metrics.precision_score(y_test,y_hat,average='micro'))
print("precision score macro")
print(metrics.precision_score(y_test,y_hat,average='macro'))
print("Recall score micro")
print(metrics.recall_score (y_test,y_hat,average='micro'))
print("Recall score")
print(metrics.recall_score (y_test,y_hat,average='macro'))
print("Jaccard micro")
print(jaccard_score(y_test,y_hat,average='micro'))
print("Jaccard macro")
print(jaccard_score(y_test,y_hat,average='macro'))



#implement binary relevance with random forest
start_time = time.time()
classifier = BinaryRelevance(
    classifier = RandomForestClassifier(),
    require_dense = [False, True]
)
classifier.fit(x_train, y_train)
print ((time.time() - start_time))
y_hat=classifier.predict(x_test)
br_f1=metrics.f1_score(y_test, y_hat, average='micro')
br_f2=metrics.f1_score(y_test, y_hat, average='macro')
br_hamm=metrics.hamming_loss(y_test,y_hat)
print('Classifier Chain micro F1-score:',round(br_f1,3))
print('Classifier Chain macro F1-score:',round(br_f2,3))
print('Classifier Chain Hamming Loss:',round(br_hamm,3))
print("Exact matching score")
print(accuracy_score(y_test,y_hat))
print("precision score micro")
print(metrics.precision_score(y_test,y_hat,average='micro'))
print("precision score macro")
print(metrics.precision_score(y_test,y_hat,average='macro'))
print("Recall score micro")
print(metrics.recall_score (y_test,y_hat,average='micro'))
print("Recall score")
print(metrics.recall_score (y_test,y_hat,average='macro'))
print("Jaccard micro")
print(jaccard_score(y_test,y_hat,average='micro'))
print("Jaccard macro")
print(jaccard_score(y_test,y_hat,average='macro'))

#implementation classifier chain with random forest
classifier = ClassifierChain(
    classifier = RandomForestClassifier(),
    require_dense = [False, True],

)
classifier.fit(x_train,y_train)

y_hat=classifier.predict(x_test)
cc_f1=metrics.f1_score(y_test, y_hat, average='micro')
cc_f2=metrics.f1_score(y_test, y_hat, average='macro')
cc_hamm=metrics.hamming_loss(y_test,y_hat)
print('Classifier Chain micro F1-score:',round(cc_f1,3))
print('Classifier Chain macro F1-score:',round(cc_f2,3))
print('Classifier Chain Hamming Loss:',round(cc_hamm,3))
print("Exact matching score")
print(accuracy_score(y_test,y_hat))
print("precision score micro")
print(metrics.precision_score(y_test,y_hat,average='micro'))
print("precision score macro")
print(metrics.precision_score(y_test,y_hat,average='macro'))
print("Recall score micro")
print(metrics.recall_score (y_test,y_hat,average='micro'))
print("Recall score")
print(metrics.recall_score (y_test,y_hat,average='macro'))
print("Jaccard micro")
print(jaccard_score(y_test,y_hat,average='micro'))
print("Jaccard macro")
print(jaccard_score(y_test,y_hat,average='macro'))
