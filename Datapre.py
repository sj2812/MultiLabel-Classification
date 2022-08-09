import csv
import re
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
import sys

ps = PorterStemmer()

directory = sys.argv[1]
mainlist = {}
listlabel={}


# a file with nltk stop words and the words which occurred more frequently
f1=open(sys.argv[2],'r')
stopwordstr=f1.read()

#Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl. If the pattern isn’t found, string is returned unchanged. repl can be a string or a function.
#so in our case : #pattern is any non-alphanumeric character. #[\w] means any alphanumeric character and is equal to the character set [a-zA-Z0-9_]
#a to z, A to Z , 0 to 9 and underscore.#so we match any non-alphanumeric character and replace it with a space . #and then we split() it which splits string by space and converts it to a list


stopwordlist=re.sub("[^\w]", " ",  stopwordstr).split()
cnt=0
id=0
for filename in os.listdir(directory):

    if filename.endswith(".html"):
        content=[]
        wordList=[]
        f=open(directory+"/"+filename,'r',encoding="utf8")
        html_doc=f.read()
        soup = BeautifulSoup(html_doc, 'html.parser')
        try:
            p=list(soup.body.children)[9]
            st=list(p.children)
            l=soup.find_all('strong')[1]
            la=list(l.parents)[0].text


            if(len(list(p.children))<35):
                continue
            s=list(p.children)[5].text+list(p.children)[35].text # takes the title and the content both

            # Print the first few characters
            #print(soup.find('div',{"class":"texte"}))
            labels = []
            contentmain=[]
            wordList = re.sub("[^\w]", " ",  s).split()
            labellist = la.split('\n')
            la=[]
            for l in labellist:
                if (l != ''):
                    labels.append(l)
            if(labels.pop(0)!='EUROVOC descriptor:'): #no eurovoc descriptors availabel then we dont add both the labels and the content to the respective list
                raise Exception("incorrect labelling")
            content=[word for word in wordList if word not in stopwordlist]
            labelsmain=[word for word in labels if word not in stopwordlist]

            content=[x.lower() for x in content]
            for w in content:
                contentmain.append(ps.stem(w))   #porter stemming
            for s in labelsmain:
                la.append(ps.stem(s))
            mainlist[id]=(list(set(contentmain)))

            listlabel[id]=labels
        except IndexError as error:
            print(error)
        except AttributeError as aerror:
            print(aerror)
        except NameError as err:
            print(err)
        except:
            print ("dk")
        id=id+1

        continue
    else:
        continue

#create two csv files one for content and one for labels
wm= csv.writer(open("content.csv", "w",encoding="utf-8"))
for key, val in mainlist.items():
    wm.writerow([key, val])

w = csv.writer(open("label.csv", "w",encoding="utf-8"))
for key, val in listlabel.items():
    w.writerow([key, val])

