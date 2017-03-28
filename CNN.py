import sys
import data_handler_org as dh

import numpy as np
from sklearn.cross_validation import train_test_split

from chainer import Chain, optimizers, Variable
import chainer.initializers as I
import chainer.links as L
import chainer.functions as F
import chainer
import chainer.computational_graph as c

class CNN(Chain):
	def __init__(self, vocab_size, embed_size, input_channel, output_channel, filter_height, filter_width, n_units, n_label):
		super(CNN, self).__init__(
			embed = L.EmbedID( vocab_size, embed_size, initialW=I.Uniform(1. / embed_size) ),
			conv1 = L.Convolution2D(input_channel, output_channel, (filter_height, filter_width)),
			l1 = L.Linear(None, n_units),
			l2 = L.Linear(n_units, n_label),
		)
		
	def __call__(self, x):
		vec = self.embed(x)
		h1 = F.max_pooling_2d(F.relu(self.conv1(vec)), 3)
		h2 = F.dropout(F.relu(self.l1(h1)))
		y = self.l2(h2)

		return y


dataset, height, vocab_size = dh.load_rt()

embed_size = 50
width = embed_size

print '---\t---\t---'
print '# of voc:\t{}'.format( vocab_size )
print 'len of doc:\t{}'.format( height )
print 'vector_dim:\t{}'.format( width )

dataset['source'] = dataset['source'].astype(np.int32)
dataset['target'] = dataset['target'].astype(np.int32)

x_train, x_test, y_train, y_test = train_test_split(dataset['source'], dataset['target'], test_size=0.15)
N_test = y_test.size #1600
N = len(x_train)
in_units = x_train.shape[1]
print '# of training data:\t{}'.format(y_train.size)
print '# of testing data:\t{}'.format(N_test)

input_channel = 1
x_train = x_train.reshape(len(x_train), input_channel, height)#width)
x_test = x_test.reshape(len(x_test), input_channel, height)#width )

n_units = 500
n_label = 2
filter_height = 3
output_channel = 50
batch_size = 64
n_epoch = 20


print 'filter size:\t{}x{}'.format(width, filter_height)
print 'batch size:\t{}\n'.format(batch_size)

model = CNN(vocab_size, embed_size, input_channel, output_channel, filter_height, width, n_units, n_label)
cf = L.Classifier(model)
optimizer = optimizers.AdaGrad()
optimizer.setup(cf)


print '---\ttrain\t---'

for epoch in xrange(1, n_epoch+1):
	print 'epoch:\t{}/{}'.format(epoch, n_epoch)

	perm = np.random.permutation(N)
	sum_train_loss = 0.0
	sum_train_accuracy = 0.0

	for i in xrange(0, N, batch_size):
		x = chainer.Variable(np.asarray(x_train[perm[i:i+batch_size]]))
		t = chainer.Variable(np.asarray(y_train[perm[i:i+batch_size]]))

		optimizer.update(cf, x, t)

		sum_train_loss += float(cf.loss.data) * len(t.data)
		sum_train_accuracy += float(cf.accuracy.data) * len(t.data)

	print 'train mean loss={}, accuracy={}'.format(sum_train_loss/N, sum_train_accuracy/N)

	sum_test_loss = 0.0
	sum_test_accuracy = 0.0

	for i in xrange(0, N_test, batch_size):
		x = chainer.Variable(np.asarray(x_test[i:i+batch_size]))
		t = chainer.Variable(np.asarray(y_test[i:i+batch_size]))

		sum_test_loss += float(cf.loss.data) * len(t.data)
		sum_test_accuracy += float(cf.accuracy.data) * len(t.data)

	print 'test mean loss={}, accuracy={}'.format(sum_test_loss/N_test, sum_test_accuracy/N_test)
	
	if epoch > 10:
		optimizer.lr *= 0.97
		print 'learning rate:\t', optimizer.lr

#output predicted result
'''
print '---\tPredicted result\t---'

learned_y = np.zeros_like(y_test)
x = Variable(np.array(x_test, dtype=np.int32))
learned_y = model(x).data

count = 0
for i in xrange(len(learned_y)):
	if np.argmax(learned_y[i]) == y_test[i]:
		count+=1
	print learned_y[i], np.argmax(learned_y[i]), y_test[i]


print 'testing accuracy:\t{}'.format(count/float(len(learned_y)))
'''
