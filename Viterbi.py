import sys


class Word:
    def __init__(self, word, ground_truth_tag):
        self.word = word
        self.ground_truth_tag = ground_truth_tag
        self.believed_tag = None
        self.potential_tags = [()]  # Defined as a list of tuples (tag_i-1, tag_i, score)


class Viterbi:
    def __init__(self, training_file):
        self.training_file = training_file
        self.tag_frequencies = {}  # Defined as {tag: frequency}
        self.bigram_probabilities = {}  # Defined as {"Ti Tj": P(Tj|Ti) = count(Ti,Tj)/count(Ti)}
        self.lexical_probabilities = {}  # Defined as {"Ti Wi": P(Wi|Ti) = count(Ti,Wi)/count(Ti)}
        self.train()
        self.count_of_correctly_labeled_tags = 0
        self.count_of_tags = 0

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

    def __viterbi(self, sentence):
        Tags = list(self.tag_frequencies.keys())
        # Initialization Step
        for tag in Tags:
            if tag == "<s>":
                continue
            word_given_tag = sentence[0].word + "/" + tag
            if word_given_tag in self.lexical_probabilities:
                tag_given_beginning = "<s>/" + tag
                sentence[0].potential_tags.append(("<s>", tag, self.lexical_probabilities[word_given_tag] *
                                                   self.bigram_probabilities[tag_given_beginning]))
            else:
                sentence[0].potenial_tags.append(("<s>", tag, 0))

        # Iteration Step
        for i in range(1, len(sentence)):
            for tag in Tags:
                if tag == "<s>":
                    continue
                word_given_tag = sentence[i].word + "/" + tag  # Time/NN
                if word_given_tag in self.lexical_probabilities:
                    max_score = 0
                    max_tag = None
                    for potential_tag in sentence[i - 1].potential_tags:  # For each tag in the previous word
                        tag_given_tag = potential_tag[1] + "/" + tag  # potential_tag[1] = previous tag
                        score = potential_tag[2] * self.lexical_probabilities[word_given_tag] * \
                                self.bigram_probabilities[tag_given_tag]  # potential_tag[2] = previous score
                        if score > max_score:
                            max_score = score
                            max_tag = potential_tag[1]
                    sentence[i].potential_tags.append((max_tag, tag, max_score))
                else:
                    sentence[i].potential_tags.append((None, tag, 0))

    def test(self, test_file):
        # Iterate over each line in the file
        with open(test_file, 'r') as f:
            for line in f:
                sentence = []
                # Iterate over each word in the line
                for phrase in line.split(" "):
                    if phrase == "\n":
                        continue
                    word = phrase.split("/")[0]
                    tag = phrase.split("/")[1]
                    sentence.append(Word(word, tag))
                self.__viterbi(sentence)

    def printAccuracy(self):
        if self.count_of_tags == 0:
            print("Accuracy not calculated yet")
        else:
            print("Accuracy: " + str(self.count_of_correctly_labeled_tags / self.count_of_tags))


def main(argv):
    training_file = argv[1]
    testing_file = argv[2]
    viterbi = Viterbi(training_file)
    viterbi.test(testing_file)
    viterbi.printAccuracy()


if __name__ == "__main__":
    # input: % python Viterbi.py POS.train POS.test
    main(sys.argv)
