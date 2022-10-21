import sys
import math


class Word:
    def __init__(self, word, ground_truth_tag):
        self.word = word
        self.ground_truth_tag = ground_truth_tag
        self.believed_tag = None
        self.potential_tags = []  # Defined as a list of tuples (tag_i-1, tag_i, score)


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
                word, tag = self.__get_word_and_tag(tag_and_word)
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
                    else:
                        tags.append(self.__get_word_and_tag(word)[1])
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
                    else:
                        tag = self.__get_word_and_tag(word)[1]

                    if tag in self.tag_frequencies:
                        self.tag_frequencies[tag] += 1
                    else:
                        self.tag_frequencies[tag] = 1

    @staticmethod
    def __get_word_and_tag(phrase):
        phrase = phrase.strip("\n")
        split = phrase.split("/")
        tag = split[len(split) - 1]
        if "|" in tag:
            tag = tag.split("|")[0]
        word = ""
        for i in range(len(split) - 1):
            word += split[i]
        return word.lower(), tag

    def __viterbi(self, sentence):
        Tags = list(self.tag_frequencies.keys())
        final_sequence = []
        # Initialization Step
        initial_tag_is_set = False
        for tag in Tags:
            if tag == "<s>":
                continue
            word_given_tag = sentence[0].word + "/" + tag
            tag_given_beginning = "<s>/" + tag
            if word_given_tag in self.lexical_probabilities and tag_given_beginning in self.bigram_probabilities:
                preliminary_score = self.lexical_probabilities[word_given_tag] * self.bigram_probabilities[
                    tag_given_beginning]
                final_score = math.fabs(math.log2(preliminary_score))
                sentence[0].potential_tags.append(("<s>", tag, final_score))
                initial_tag_is_set = True
            else:
                sentence[0].potential_tags.append(("<s>", tag, 0))
        if not initial_tag_is_set:
            sentence[0].potential_tags.append(("<s>", "NN", 1))

        # Iteration Step
        for i in range(1, len(sentence)):
            max_score = 0
            max_tag = None
            valueSet = False
            for tag in Tags:
                if tag == "<s>":
                    continue
                word_given_tag = sentence[i].word + "/" + tag  # Time/NN
                if word_given_tag in self.lexical_probabilities:
                    previous_word = sentence[i - 1]
                    max_tag, max_score = self.__get_max_tag_and_score(tag, previous_word)
                    if max_score == 0:
                        sentence[i].potential_tags.append((None, tag, 0))
                        continue
                    final_score = math.fabs(math.log2(max_score * self.lexical_probabilities[word_given_tag]))
                    sentence[i].potential_tags.append((max_tag, tag, final_score))
                    valueSet = True
                else:
                    sentence[i].potential_tags.append((None, tag, 0))

            if not valueSet:
                sentence[i].potential_tags.remove((None, 'NN', 0))
                sentence[i].potential_tags.append(
                    (self.__highest_scoring_tag(sentence[i - 1].potential_tags)[1], "NN", 1))

        # Sequence Identification
        max_tag = self.__highest_scoring_tag(
            sentence[len(sentence) - 1].potential_tags)
        sentence[len(sentence) - 1].believed_tag = max_tag[1]
        final_sequence.append(max_tag[1])
        for i in range(len(sentence) - 2, -1, -1):
            sentence[i].believed_tag = max_tag[0]
            final_sequence.append(max_tag[0])
            max_tag = self.__get_tag_obj_for_tag(sentence[i].believed_tag, sentence[i])

        # final_sequence.append("<s>")
        final_sequence.reverse()

        return final_sequence

    @staticmethod
    def __get_tag_obj_for_tag(tag, word):
        for tag_obj in word.potential_tags:
            if tag_obj[1] == tag:
                return tag_obj

    @staticmethod
    def __highest_scoring_tag(potential_tags):
        max_score = 0
        max_tag = None
        for tag in potential_tags:
            if tag[2] > max_score:
                max_score = tag[2]
                max_tag = tag
        return max_tag

    def __get_max_tag_and_score(self, my_tag, previous_word):
        max_tag = None
        max_score = 0
        for previous_tag in previous_word.potential_tags:
            tag_given_tag = previous_tag[1] + "/" + my_tag
            if tag_given_tag in self.bigram_probabilities:
                score = previous_tag[2] * self.bigram_probabilities[tag_given_tag]
                if score > max_score:
                    max_score = score
                    max_tag = previous_tag[1]
        return max_tag, max_score

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
                    word, tag = self.__get_word_and_tag(phrase)
                    sentence.append(Word(word, tag))

                predicted = self.__viterbi(sentence)

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
