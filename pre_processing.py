import sys
import re
import os.path
import unicodedata
import multiprocessing
import json

import pandas as pd
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from collections import defaultdict, OrderedDict

class Preprocessing():
	def __init__(self):
		self.dataset = pd.read_csv('uci-news-aggregator.csv')
		self.titles = [title for title in self.dataset['TITLE']]
		self.true_label = [label for label in self.dataset['CATEGORY']]
		self.unique_label = np.unique(self.true_label)
		self.selected_labels = []
		self.corpus_freq = defaultdict(int)

	def clean_str(self, string):
		string = re.sub(r"\\", "", string)    
		string = re.sub(r"\'", "", string)    
		string = re.sub(r"\"", "", string)
		return string.strip().lower()

	def lemmatize(self):
		# Only create stemmed word file if no file exists
		if (not os.path.exists('lemmatized.txt') or not os.path.exists('lemmatized_sentences.txt')):
			print("Creating text files to extract from......")
			with open('lemmatized.txt', 'w') as file, open('lemmatized_sentences.txt', 'w') as file1, open('label.txt', 'w') as labels:
				for ind, title in enumerate(self.titles):
					temp_title = []
					# split title into vector of words
					if "\xc2" not in title:
						labels.write(str(list(self.unique_label).index(self.true_label[ind])) + '\n')
						title = title.replace("\xe2", " ")
						for word in title.split():
							word = word.lower().rstrip('?:!.,;')
							word = clean_str(word)
							word = ''.join(c for c in word if c.isalpha())
							if word not in stopword_set and 'http://' not in word and 'www' not in word:
								if word_pattern.match(word):
									# Add stemmed words to text file
									try:
										# word_stem = ps.stem(word).rstrip("'")
										word_temp = wl.lemmatize(word).rstrip("'")
										file.write(word_temp + '\n')
										temp_title.append(word_temp)
									except UnicodeDecodeError:
										print(word)
										# print(ps.stem(word).rstrip("'"))
										print("=================\n")
						# Write titles to stemmed_sentences.txt
						file1.write(' '.join(temp_title) + '\n')
		else:
			print ("lemmanized text files already present.")

	def process_data(self, process):
		if process:
			# List of all stemmed words (bag of words)
			all_words = []
			feature = OrderedDict()
			with open('lemmatized.txt', 'r') as file:
				for word in file:
					word = word.strip('\n')
					all_words.append(word)
					self.corpus_freq[word] += 1.0
					feature[word] = 0

			# Contains word freq dictionary for each title
			titles_doc_vector = []
			print(feature[1843])
			with open('lemmatized_sentences.txt', 'r') as file1:
				for title in file1:
					new_doc_vec = defaultdict(int)
					for word in title.split():
						new_doc_vec[word] += 1
					titles_doc_vector.append(new_doc_vec)

			# Set of all unique stemmed words
			vocab_set = list(set(all_words))
			print("length of all words (including repeats) is: " + str(len(all_words)))
			print("length of vocab list is " +str(len(vocab_set)))
			print("length of titles list is " + str(len(titles_doc_vector)))

			freq_count = defaultdict(int)
			for key, val in self.corpus_freq.items():
				freq_count[val] += 1
			print("Number of words with freq = 1: " + str(freq_count[1]))
			print("Number of words with freq = 2: " + str(freq_count[2]))
			print("Number of words with freq = 3: " + str(freq_count[3]))
			print("Number of words with freq = 4: " + str(freq_count[4]))
			print("Number of words with freq = 5: " + str(freq_count[5]))
			print("Number of words with freq = 6: " + str(freq_count[6]))
			print("Number of words with freq = 7: " + str(freq_count[7]))
			print("Number of words with freq = 8: " + str(freq_count[8]))
			print("Number of words with freq = 9: " + str(freq_count[9]))
			print("Number of words with freq = 10: " + str(freq_count[10]))

			freq_descending = sorted(freq_count, reverse=True)

			#removing unnecessary
			for key, val in self.corpus_freq.items():
				if (val >= 0 and val <= 20) or val in freq_descending[0:200]:
					del self.corpus_freq[key]
					del feature[key]
			print('\n')
			print('finished processing')
			print("length of features is " +str(len(feature)))
			print("length of features is " +str(len(self.corpus_freq)))
			print("length of examples is " + str(len(titles_doc_vector)))

			jobs = []
			for i in range(1,5):
				p = multiprocessing.Process(target=self.run_thread, args=(titles_doc_vector[(i-1)*20000:i*20000], feature, (i-1)*20000))
				jobs.append(p)
				p.start()

	def run_thread(self, doc_vector, feature, start):
		feature_vector = OrderedDict()
		counter = 0
		print("start :" + str(start))
		for freq_dict in doc_vector:
			counter += 1
			instance = feature.copy()
			for word, freq in freq_dict.items():
				if word in self.corpus_freq and self.corpus_freq[word] != 0:
					instance[word] = freq / self.corpus_freq[word]
			feature_vector[counter] = instance.values()
			if counter == 1000:
				start += 1
				with open('feature_json_multi/feature_vector' + str(start) + '.json', 'w') as fp:
					json.dump(feature_vector, fp)
					feature_vector.clear()
					print('finished ' + str(start) + ' json file')
					counter = 0