import nltk
import numpy as np
import gensim
from itertools import product, chain
from collections import Counter, deque

def gen_vec( sentences, dim=50 ):
	#model = gensim.models.Word2Vec(sentences, size=dim, min_count=1)
	model = gensim.models.Word2Vec.load_word2vec_format('/Users/joli/googleVector/model.bin', binary=True)

	return model

def load_rt():
	import string
	printable = set(string.printable)

	neg = open('./rt-data/rt-polarity.neg').readlines()
	pos = open('./rt-data/rt-polarity.pos').readlines()

	neg = ['{}::0::0'.format(filter(lambda x: x in printable, i).strip()) for i in neg]
	pos = ['{}::0::1'.format(filter(lambda x: x in printable, i).strip()) for i in pos]

	return load_data( neg+pos, dim=50 )


def load_data( articles, dim=50 ):
	import os
	import nltk

	docs = []
	keywords = []
	labels = []
	
	print 'preprocess\t\t---'

	for article in articles:
		shards = article.split('::')
		
		docs.append( nltk.word_tokenize(shards[0]) )
		keywords.append( shards[1] )
		labels.append( shards[2] )

	docs, max_len = pad_sentences(docs)


	#print 'word2vec\t---'
	#model = gen_vec(docs, dim)
	voc = {x: i for i, x in enumerate( [x[0] for x in Counter(chain(*docs)).most_common()] )}
	
	print 'voc_size: ', len(voc)
	print 'finish preprocess\t---\n'

	source = []
	keyword = []

	for doc in docs:
		doc_vec = []

		for word in doc:
			vec = voc[word]
			doc_vec.append(vec)
			'''
			try:
				vec = model[word]
			except KeyError:
				vec = model.seeded_vector(word)
				print 'KeyError:\t', word

			doc_vec.extend(vec)
			dim = len(vec)
			'''
		source.append(doc_vec)

	'''
	for k in keywords:
		try:
			vec = model[k]
		except KeyError:
			vec = model.seeded_vector(k)
		#	print 'KeyError:\t', k
		keyword.append(vec)
	'''

	dataset = {}
	dataset['source'] = np.array(source)
	dataset['target'] = np.array(labels)

	return dataset, max_len, len(voc)

def pad_sentences(docs, padding_word='<PAD/>'):
	max_len = max( [len(doc) for doc in docs])
	return [doc + ([padding_word] * (max_len-len(doc))) for doc in docs], max_len
	

