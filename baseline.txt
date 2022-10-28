import sys

from Viterbi import Viterbi
from Viterbi import Word


class Baseline(Viterbi):
    def __init__(self, training_file):
        super().__init__(training_file)
        self.train()

    def __baseline(self, sentence):
        predicted = []
        for word in sentence:
            predicted.append(self.__get_most_frequent_tag(word))
        return predicted

    def test(self, test_file):
        # Clear out file
        open("POS.test.out", 'w').close()
        # Iterate over each line in the file
        with open(test_file, 'r') as f:
            for line in f:
                sentence = []
                # Iterate over each word in the line
                for phrase in line.split(" "):
                    if phrase == "\n":
                        continue
                    word, tag = self._get_word_and_tag(phrase)
                    sentence.append(Word(word, tag))

                predicted = self.__baseline(sentence)

                # Print the predicted tags to a file
                with open("POS.test.out", 'a') as out:
                    for i in range(len(sentence)):
                        out.write(sentence[i].word + "/" + predicted[i] + " ")
                    out.write("\n")
                # Update the accuracy
                for i in range(len(sentence)):
                    if sentence[i].ground_truth_tag == predicted[i]:
                        self.count_of_correctly_labeled_tags += 1
                    self.count_of_tags += 1

    def __get_most_frequent_tag(self, word):
        max_tag = None
        max_count = 0
        tags = self.tag_frequencies.keys()
        for tag in tags:
            # Get word from lexical probability
            word_given_tag = word.word + "/" + tag
            if word_given_tag in self.lexical_probabilities:
                count = self.lexical_probabilities[word_given_tag]
                if count > max_count:
                    max_count = count
                    max_tag = tag
        if max_tag is None:
            return "NN"
        return max_tag


def main(argv):
    training_file = argv[1]
    testing_file = argv[2]
    baseline = Baseline(training_file)
    baseline.test(testing_file)
    baseline.printAccuracy()


if __name__ == "__main__":
    main(sys.argv)
