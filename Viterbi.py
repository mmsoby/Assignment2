import sys


class Word:
    def __init__(self, word, tag):
        self.word = word
        self.ground_truth_tag = tag
        self.believed_tag = None
        self.potential_tags = []


class Viterbi:
    def __init__(self, training_file):
        self.training_file = training_file
        self.tag_frequencies = {}  # Defined as {tag: frequency}
        self.bigram_probabilities = {}  # Defined as {"Ti Tj": P(Tj|Ti) = count(Ti,Tj)/count(Ti)}
        self.lexical_probabilities = {}  # Defined as {"Ti Wi": P(Wi|Ti) = count(Ti,Wi)/count(Ti)}
        self.train()
        self.accuracy = -1

    def train(self):
        self.__get_tag_frequencies()
        self.__get_bigram_probabilities()
        self.__get_lexical_probabilities()

    def __get_lexical_probabilities(self):
        lexical_counts = {}
        # Grab tags line by line
        with open(self.training_file, 'r') as f:
            for line in f:
                # Calculate lexical Counts
                for tag_and_word in line.split(" "):
                    if tag_and_word == "\n":
                        continue
                    if tag_and_word in lexical_counts:
                        lexical_counts[tag_and_word] += 1
                    else:
                        lexical_counts[tag_and_word] = 1
            # Calculate lexical probabilities
            for tag_and_word in lexical_counts:
                tag = tag_and_word.split("/")[1]
                self.lexical_probabilities[tag_and_word] = lexical_counts[tag_and_word] / self.tag_frequencies[tag]

    def __get_bigram_probabilities(self):
        bigram_counts = {}  # Defined as {"Ti Tj": count(Ti,Tj)}
        # Grab tags line by line
        with open(self.training_file, 'r') as f:
            for line in f:
                tags = []
                for word in line.split(" "):
                    if word == "\n":
                        continue
                    tags.append(word.split('/')[1])
                # Calculate bigram Counts
                for i in range(len(tags)):
                    if i == 0:
                        bigram = "<s>/" + tags[i]
                    else:
                        bigram = tags[i - 1] + "/" + tags[i]
                    if bigram in bigram_counts:
                        bigram_counts[bigram] += 1
                    else:
                        bigram_counts[bigram] = 1

        # Calculate bigram probabilities
        for bigram in bigram_counts:
            tag = bigram.split("/")[0]
            self.bigram_probabilities[bigram] = bigram_counts[bigram] / self.tag_frequencies[tag]

    def __get_tag_frequencies(self):
        # Get frequency of each tag in the training data
        self.tag_frequencies["<s>"] = 0
        with open(self.training_file, 'r') as f:
            for line in f:
                self.tag_frequencies["<s>"] += 1
                for word in line.split(" "):
                    if word == "\n":
                        continue
                    tag = word.split('/')[1]
                    if tag in self.tag_frequencies:
                        self.tag_frequencies[tag] += 1
                    else:
                        self.tag_frequencies[tag] = 1

    def test(self, test_file):
        pass

    def printAccuracy(self):
        if self.accuracy == -1:
            print("Accuracy not calculated yet")
        else:
            print("Accuracy: " + str(self.accuracy))


def main(argv):
    training_file = argv[1]
    testing_file = argv[2]
    viterbi = Viterbi(training_file)
    viterbi.test(testing_file)
    viterbi.printAccuracy()


if __name__ == "__main__":
    # input: % python Viterbi.py POS.train POS.test
    main(sys.argv)
