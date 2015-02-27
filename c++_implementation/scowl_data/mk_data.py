#python3.4.2
import random
import string
import numpy as np

def get_words(length):
	"""returns array of valid british words with given length
	"""
	
	path = "../../data/scowl/"
	#initalize containing array
	words = []
	
	#read in the scowl data file
	file = open(path + "english.55")
	for word in file:
		try:                        #skip non-ascii words
		    word.encode('ascii')
		except UnicodeDecodeError:
		    continue
		
		word=word[:-1]              #removing \n
		
		if not word.islower():      #skip names
		    continue
		if not word.isalpha():      #skip apostrophes, hyphens etc.
		    continue
		if len(word) == length:
			words.append(word)
	
	file.close()#.encode('iso-8859-1') as dic:
	
	random.shuffle(words)
	return words


def str_generator(length):
    """generates random strings from lowercase alphabet for given length
    """
    
    chars = string.ascii_lowercase
    while True:
        yield ''.join(random.SystemRandom().choice(chars) for _ in range(length))


def make_data(words, size):
	"""returns nparray containing size valid words and size notwords
	"""
	
	assert size <= len(words)
	#initalize array for instances + class
	data = np.zeros( (2*size, len(words[0]) + 1) )
	
	eps = 0.05
	
	#fill one half with existing words...
	for i in range(size):
	    data[i][-1] = 0                                 #label 0 will be generated
	    for j in range(len(words[i])):
			data[i][j] = ord(words[i][j]) - ord('a')    #code: 'a' = 0
	
	#...the other half with random character strings (notwords)
	gen = str_generator(len(words[0]))
	for i in range(size, 2*size):
	    data[i][-1] = 1
	    notword = next(gen)
	    while notword in words:
	        notword = next(gen)
	    
	    for j in range(len(notword)):
			data[i][j] = ord(notword[j]) - ord('a')

	#permutate and split into words + labels
	data = np.random.permutation(data)
	return data[:,:-1], data[:,-1]


def mean_word_length():
    """calculates the (weighted) mean word length
    """
    
    N = 50
    N_words = []
    for length in range(N):
        words = get_words(length)
        N_words.append(len(words))
    
    N_words = np.array(N_words)/sum(N_words)
    return sum(N_words[i]*i for i in range(N))


if __name__ == '__main__':
	#read in words
	length = raw_input("Choose length of words (default = 5): ")
	if length == '':
	    length = 5
	dic = get_words(int(length))
	print(len(dic), "words of length", length, "loaded.")
	
	# set up data
	data, labels = make_data( dic, len(dic) )
	# split into training data (80 %) and test_data (20%)
	split_indx 		= int( np.floor( 0.8*data.shape[0] ) )
	train_data 		= data[:split_indx]
	train_labels 	= labels[:split_indx]
	test_data 		= data[split_indx+1:]
	test_labels 	= labels[split_indx+1:]
	
	print("eeeaasy")
	
	#save data to files
	path="original/"
	np.savetxt(path + "data_train.out", 	train_data, 	fmt='%f')
	np.savetxt(path + "labels_train.out", 	train_labels, 	fmt='%i')
	np.savetxt(path + "data_test.out", 		test_data, 		fmt='%f')
	np.savetxt(path + "labels_test.out", 	test_labels, 	fmt='%i')
	print("'.out' files save to " + path)
