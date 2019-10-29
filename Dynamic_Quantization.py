"""
Dynamic quantization on an lstm word language model
"""

import os
from io import open
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
	"""
		Container module with an encoder, a recurrent module and a decoder.
	"""

	def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
		super(LSTMModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(ntoken, ninp)
		self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
		self.decoder = nn.Linear(nhid, ntoken)

		self.init_weights()
		self.nhid = nhid
		self.nlayers = nlayers

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden):
		emb = self.drop(self.encoder(input))
		output, hidden = self.rnn(emb, hidden)
		output = self.drop(output)
		decoded = self.decoder(output)
		return decoded, hidden

	def init_hidden(self, bsz):
		weight = next(self.parameters())
		return (weight.new_zeros(self.nlayers, bszm,self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid))

class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []

	def add_word(self, word):
		if(word not in self.word2idx):
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)

class Corpus(object):
	def __init__(self, path):
		self.dictionary = Dictionary()
		self.train = self.tokenize(os.path.join(path, 'train.txt'))
		self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
		self.test = self.tokenize(os.path.join(path, 'test.txt'))

	def tokenize(self, path):
		"""Tokenizes a text file"""
		assert os.path.exists(path)

		# Adding words to the dictionary
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				words  = line.split() + ['<eos>']
				for word in words:
					self.dictionary.add_word(word)


		# Tokenize file content
		with open(path, 'r', encoding="utf-8") as f:
			idss = []
			for line in f:
				words = line.split() + ['<eos>']
				ids = []
				for word in words:
					ids.append(self.dictionary.word2idx[word])
				idss.append(torch.tensor(ids).type(torch.int64))
			ids = torch.cat(idss)

		return ids

model_data_path = "C:\\Users\\RISHAV\\Documents\\kaggle_days\\Text Classification\\data"
corpus = Corpus(model_data_path + '\\wikitext-2')

ntokens = len(corpus.dictionary)
model = LSTMModel(ntoken = ntokens, ninp=512, nhid=256, nlayers=5,)
model.load_state_dict(torch.load(model_data_path + '\\word_language_model_quantize.pth', map_location = torch.device('cpu')))
model.eval()
print(model)

input_ = torch.randint(ntokens, (1,1), dtype=torch.long)
hidden = model.init_hidden(1)
temperature = 1.0
num_words = 1000

with open(model_data_path + '\\out.txt', 'w') as outf:
	with torch.no_grad():
		for i in range(num_words):
			output, hidden = model(input_, hidden)
			word_weights = output.squeeze().div(temperature).exp().cpu()
			word_idx = torch.multinomial(word_weights, 1)[0]
			input_.fill_(word_idx)

			word = corpus.dictionary.idx2word[word_idx]
			outf.write(str(word.encode('utf-8')) + ('\n' if i % 20 == 19 else ' '))

			if(i%100==0):
				print('| Generated {}/{} words'.format(i, 1000))

with open(model_data_path + '\\out.txt', 'r') as outf:
	all_output = outf.read()
	print(all_output)


# Helper functions
bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

def batchify(data, bsz):
	nbatch = daat.size(0) // bsz
	data = data.narrow(0, 0, nbatch * bsz)
	return data.view(bsz, -1).t().contiguous()

test_data = batchify(corpus.test, eval_batch_size)

def get_batch(source, i):
	seq_len = min(bptt, len(source)-1-i)
	data = source[i:i+seq_len]
	target = source[i+1:i+1+seq_len].view(-1)
	return data, target

def repackage_hidden(h):
	if(isinstance(h, torch.Tensor)):
		return h.detach
	else:
		return tuple(repackage_hidden(v) for v in h)

def evaluate(model_, data_source):
	model_.eval()
	total_loss = 0
	hidden = model_.init_hidden(eval_batch_size)
	with torch.no_grad():
		for i in range(0, data_source.size(0) - 1, bptt):
			data, targets = get_batch(data_source, i)
			output, hidden = model_(data, hidden)
			hidden = repackage_hidden(hidden)
			output_flat = output.view(-1, ntokens)
			total_loss += len(data) * criterion(output_flat, targets).item()

	return total_loss / (len(data_source) - 1)


import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
print(quantized_modeln)
