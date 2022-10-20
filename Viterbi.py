import sys
import numpy as np


class Viterbi:
    def __init__(self, training_file):
        self.training_file = training_file
        self.training_sentences, self.training_tags = Viterbi.preprocess(self.training_file)

    @staticmethod
    def preprocess(file):
        temp_sentences, temp_tags = [[]], [[]]
        # Get the training data line by line
        with open(file, 'r') as f:
            for line in f:
                # Add a tag and word for the start of the sentence
                temp_sentences.append('<s>')
                temp_tags.append('<s>')

                # Split the line into words and tags
                words_tags = line.split()

                # Split the words and tags into separate lists
                words = [word_tag.split('/')[0] for word_tag in words_tags]
                tags = [word_tag.split('/')[1] for word_tag in words_tags]

                # Append the lists to the sentences and tags lists
                temp_sentences.append(words)
                temp_tags.append(tags)

        return temp_sentences, temp_tags

    def train(self):
        # Gather the necessary tag frequencies and bigram probabilities
        pass

    def __viterbi(self, sentence):
        pass

    def test(self, test_file):
        testing_sentences, testing_tags = Viterbi.preprocess(test_file)

        # From the start of this sentence to the end of this sentence
        temp = np.array(testing_sentences)
        sentences = np.split(temp, np.where(temp == '<s>')[0])
        sentences = [item[1:] for item in sentences if len(item) > 1]

        # Run viterbi on each sentence
        for sentence in sentences:
            self.__viterbi(sentence.tolist()[0])

    def printAccuracy(self):
        pass


def main(argv):
    training_file = argv[1]
    testing_file = argv[2]
    viterbi = Viterbi(training_file)
    viterbi.train()
    viterbi.test(testing_file)
    viterbi.printAccuracy()


if __name__ == "__main__":
    # input: % python Viterbi.py POS.train POS.test
    main(sys.argv)
