# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

from django.http import HttpResponseRedirect, HttpResponse

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.views.decorators.csrf import csrf_exempt

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier																																																																																	
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import csv
import re

# Create your views here.
@api_view(["POST", "GET"])
@csrf_exempt

def classify(request):
	text = request.POST['text']
	
	df = pd.read_table("/home/dax/headnotes/training/007.txt", sep='\t', header=None)
	df_test = pd.read_table("/home/dax/headnotes/test/test_007_trimmed.txt", sep='\t', header=None)
	# delimiter = '\t'
	# delimiter = delimiter.encode('utf-8')
	# with open("/home/dax/headnotes/training/007.txt") as f:
	# 	reader = csv.reader(f, delimiter=delimiter)
	# 	d = list(reader)

	# with open("/home/dax/headnotes/test/test_007_trimmed.txt") as f:
	# 	reader = csv.reader(f, delimiter=delimiter)
	# 	d_test = list(reader)

	y_train = df[0]
	x_train = df[1]
	

	x_test = df_test[0]
	#print(x_test[0])
	#x_test = df_test[1]

	count_vect = CountVectorizer()
	x_train_counts = count_vect.fit_transform(x_train)
	x_train_counts.shape

	tfidf_transformer = TfidfTransformer()
	x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
	x_train_tfidf.shape

	clf = MultinomialNB().fit(x_train_tfidf, y_train)
	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
	
	text_clf = text_clf.fit(x_train, y_train)
	predict = text_clf.predict(x_test)

	#type(x_test)
	print predict

	predictAns = text_clf.predict(("",text))

	answer = {}
	answer["classify"] = predict[1]

	# test = {}
	# for p in range(len(predict)):
	# 	test[predict] = x_test[p]

	#print(test)
	return Response(answer)

@api_view(["POST", "GET"])
@csrf_exempt

def classify2(request):
	text = request.POST['text']

	#d = pd.read_table("/home/dax/headnotes/training/007.txt", sep='\t', header=None)
	#d_test = pd.read_table("/home/dax/headnotes/test/007.txt", sep='\t', header=None)
	delimiter = '\t'
	delimiter = delimiter.encode('utf-8')
	with open("/home/dax/headnotes/training/007.txt") as f:
		reader = csv.reader(f, delimiter=delimiter)
		d = list(reader)

	for i in range(len(d)):
		d[i][1]=re.sub('\d+','',d[i][1])
		d[i][1]=re.sub('\(',' ',d[i][1])
		d[i][1]=re.sub('\)',' ',d[i][1])
		d[i][1]=re.sub('\?',' ',d[i][1])
		d[i][1]=re.sub('i. e.','',d[i][1])
		d[i][1]=re.sub('\.',' ',d[i][1])
		d[i][1]=re.sub(',',' ',d[i][1])
		d[i][1]=re.sub(':',' ',d[i][1])
		d[i][1]=re.sub(';',' ',d[i][1])
		d[i][1]=re.sub('  ',' ',d[i][1])
		d[i][1]=re.sub('-',' ',d[i][1])
		d[i][1]=re.sub('--',' ',d[i][1])
		d[i][1]=re.sub('  ',' ',d[i][1])
		d[i][1]=re.sub(' for ',' ',d[i][1])
		d[i][1]=re.sub(' and ',' ',d[i][1])
		d[i][1]=re.sub(' to ',' ',d[i][1])
		d[i][1]=re.sub(' an ',' ',d[i][1])
		d[i][1]=re.sub(' in ',' ',d[i][1])
		d[i][1]=re.sub(' the ',' ',d[i][1])
		#d[i][1]=re.sub('The ',' ',d[i][1])
		d[i][1]=re.sub(' of ',' ',d[i][1])
		d[i][1]=re.sub(' is ',' ',d[i][1])
		d[i][1]=re.sub(' it ',' ',d[i][1])
		#d[i][1]=re.sub('A ','',d[i][1])
		d[i][1]=re.sub(' if ','',d[i][1])
		d[i][1]=re.sub(' are ','',d[i][1])
		d[i][1]=re.sub(' a ','',d[i][1])


	with open("/home/dax/headnotes/test/007.txt") as f:
		reader = csv.reader(f, delimiter=delimiter)
		d_test = list(reader)
	for i in range(len(d_test)):
		d_test[i][1]=re.sub('\d+','',d_test[i][1])
		d_test[i][1]=re.sub('\(','',d_test[i][1])
		d_test[i][1]=re.sub('\)','',d_test[i][1])
		d_test[i][1]=re.sub('\?',' ',d_test[i][1])
		d_test[i][1]=re.sub('i. e.','',d_test[i][1])
		d_test[i][1]=re.sub('\.',' ',d_test[i][1])
		d_test[i][1]=re.sub(',',' ',d_test[i][1])
		d_test[i][1]=re.sub(':',' ',d_test[i][1])
		d_test[i][1]=re.sub(';',' ',d_test[i][1])
		d_test[i][1]=re.sub('  ',' ',d_test[i][1])
		d_test[i][1]=re.sub('-',' ',d_test[i][1])
		d_test[i][1]=re.sub('--',' ',d_test[i][1])
		d_test[i][1]=re.sub('  ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' for ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' and ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' to ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' an ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' in ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' the ',' ',d_test[i][1])
		#d_test[i][1]=re.sub('The ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' of ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' is ',' ',d_test[i][1])
		d_test[i][1]=re.sub(' it ',' ',d_test[i][1])
		#d_test[i][1]=re.sub('A ','',d_test[i][1])
		d_test[i][1]=re.sub(' if ','',d_test[i][1])
		d_test[i][1]=re.sub(' are ','',d_test[i][1])
		d_test[i][1]=re.sub(' a ','',d_test[i][1])

		text=re.sub('\d+','',text)
		text=re.sub('\(','',text)
		text=re.sub('\)','',text)
		text=re.sub('\?',' ',text)
		text=re.sub('i. e.','',text)
		text=re.sub('\.',' ',text)
		text=re.sub(',',' ',text)
		text=re.sub(':',' ',text)
		text=re.sub(';',' ',text)
		text=re.sub('  ',' ',text)
		text=re.sub('-',' ',text)
		text=re.sub('--',' ',text)
		text=re.sub('  ',' ',text)
		text=re.sub(' for ',' ',text)
		text=re.sub(' and ',' ',text)
		text=re.sub(' to ',' ',text)
		text=re.sub(' an ',' ',text)
		text=re.sub(' in ',' ',text)
		text=re.sub(' the ',' ',text)
		#text=re.sub('The ',' ',d_test[i][1])
		text=re.sub(' of ',' ',text)
		text=re.sub(' is ',' ',text)
		text=re.sub(' it ',' ',text)
		#d_test[i][1]=re.sub('A ','',text)
		text=re.sub(' if ','',text)
		text=re.sub(' are ','',text)
		text=re.sub(' a ','',text)


	x_train=[]
	y_train=[]

	for i in range(len(d)):
		y_train.append(d[i][0])
		x_train.append(d[i][1])



	x_test=[]
	y_test=[]

	for i in range(len(d_test)):
		y_test.append(d_test[i][0])
		x_test.append(d_test[i][1])

	from nltk.stem.snowball import SnowballStemmer
	stemmer = SnowballStemmer("english")
	for i in range(len(d)):
		temp=d[i][1].split()
		for j in range(len(temp)):
			temp[j]=stemmer.stem(temp[j])
		temp.reverse()
		temp=" ".join(temp)
		d[i][1]=temp+" "

	from nltk.stem.snowball import SnowballStemmer
	stemmer = SnowballStemmer("english")
	for i in range(len(d_test)):
		temp=d_test[i][1].split()
		for j in range(len(temp)):
			temp[j]=stemmer.stem(temp[j])
		temp=" ".join(temp)
		d_test[i][1]=temp+" "


	count_vect = CountVectorizer()
	x_train_counts = count_vect.fit_transform(x_train)

	tfidf_transformer = TfidfTransformer()
	x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
	x_train_tfidf.shape

	clf=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-4, n_iter=50, random_state=42)
	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-4, n_iter=50, random_state=42)),])

	parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-4, 1e-5),}

	gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
	gs_clf = gs_clf.fit(x_train, y_train)


	# text_clf = text_clf.fit(x_train, y_train)
	predict = gs_clf.predict(x_test)

	np.mean(predict == y_test)

	predictAns = gs_clf.predict(("", text))

	answer = {}
	answer["classify"] = predictAns[1]

	test = {}
	for p in range(len(predict)):
		test[predict] = x_test[p]
	#answer["testResult"]

	print(test)

	return Response(answer)



