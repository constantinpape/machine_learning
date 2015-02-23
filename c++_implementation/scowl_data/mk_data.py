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
    with open(path + "english.70", encoding='iso-8859-1') as dic:
        for word in dic:
            try:                        #skip non-ascii words
                word.encode('ascii')
            except UnicodeEncodeError:
                continue
            
            word=word[:-1]              #removing \n
            
            if not word.islower():      #skip names
                continue
            if not word.isalpha():      #skip apostrophes, hyphens etc.
                continue
            if len(word) == length:
                words.append(word)
    
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
    data = np.zeros([2*size, len(words[0]) + 1], dtype=int)
    
    #fill one half with existing words...
    for i in range(size):
        data[i][-1] = 1                                 #label
        for j in range(len(words[i])):
            data[i][j] = ord(words[i][j]) - ord('a')    #code: 'a' = 0
    
    #...the other half with random character strings (notwords)
    gen = str_generator(len(words[0]))
    for i in range(size, 2*size):
        data[i][-1] = 0
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
    length = input("Choose length of words (default = 5): ")
    if length == '':
        length = 5
    words = get_words(int(length))
    print(len(words), "words of length", length, "loaded.")
    
    #set up training data
    size = input("Choose size of training data (default = 3000): ")
    if size == '':
        size = 3000
    train_data, train_labels = make_data(words, int(size))
    test_data, test_labels = make_data(words, len(words))
    print("easy")
    
    #save data to files
    path="original/"
    np.savetxt(path + "data_train.out", train_data, fmt='%i')
    np.savetxt(path + "labels_train.out", train_labels, fmt='%i')
    np.savetxt(path + "data_test.out", test_data, fmt='%i')
    np.savetxt(path + "labels_test.out", test_labels, fmt='%i')
    print("'.out' files save to " + path)
    
    