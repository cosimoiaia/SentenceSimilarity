#!/usr/bin/env python 


##########################################
#
# SentenceSimilarity.py: A Simple python script to calculate similarities between vectors of sentences on a given dataset.
#                         This will also be an attempt to implement a continuous training on the Doc2Vec model from gensim.
#
# Author: Cosimo Iaia <cosimo.iaia@gmail.com>
# Date: 04/05/2018
#
# This file is distribuited under the terms of GNU General Public
#
#########################################


from __future__ import print_function
import numpy as np
from gensim import models
import argparse

FLAGS = None


def getSim(model, corpus, sentence):
	sentence = sentence.rstrip('\n')
	vector = model.infer_vector(sentence)

	sims = model.docvecs.most_similar(positive=[vector], topn=FLAGS.maxres)

	res = []
	for idx, sim in sims:
		res.append([corpus[idx], sim])

	return res

def scrubData(path):
	with open(path, 'r') as f:
		lines = f.readlines()

	ds = []
	for l in lines:
		d = l.rstrip('\n').split('\t')
		for x in d: ds.append(x)

	return ds

def labelCorpus(corpus):
	c = []
	for i, s in enumerate(corpus):
		c.append(models.doc2vec.LabeledSentence(s, [i]))
	return c

def main():
	path = FLAGS.dataset

	# Prepare the data:
	ds = scrubData(path)
	corpus = labelCorpus(ds)


	# Build and train the model
	m = models.Doc2Vec(min_count=1, window=10, size=10, sample=1e-2,negative=3, workers=4, iter=FLAGS.epochs)
	m.build_vocab(corpus)
	print("------ Training the model --------")
	m.train(corpus, epochs=m.iter, total_examples=m.corpus_count)


	# Let's test it...
	result = getSim(m, corpus, "What about the Spanish Inquisition?")

	print(result)

	# ...and play with it.

	try:
		import readline

		sentence = raw_input('Insert sentence to compare> ')

		result = getSim(m, corpus, sentence)

		for txt,sim in result:
			print(txt[0])
			print('Similarity: ', sim)

	except EOFError:
		print("Ok, bye")
		return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python Service to calculate sentence similarities')
    parser.add_argument('--dataset', type=str, required=True, default='', help='Path to the dataset file')
    parser.add_argument('--maxres', type=int, default='5', help='How many similar sentences we show')
    parser.add_argument('--epochs', type=int, default='10', help='How many epochs to train')
    FLAGS = parser.parse_args()
    main()


