import numpy as np


class Scoring:
    def __init__(self, answers, labels, isregress=False):
        self.givenAnswers = answers
        # self.trueAnswers = labels.to_list()
        self.trueAnswers = labels

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.TF = ''

        self.count = len(answers)
        if isregress:
            self.set_probability_to_class()
        else:
            self.prepare_given_answers()

        self.set_answer_category()

    def prepare_given_answers(self):
        given_proba_one = []

        for i in range(self.count):
            given_proba_one.append(self.givenAnswers[i][1])

        self.givenAnswers = given_proba_one

    def set_probability_to_class(self):
        for i in range(self.count):
            if self.givenAnswers[i] < 0.5:
                self.givenAnswers[i] = 0
            else:
                self.givenAnswers[i] = 1

    def set_category_regress(self):
        for i in range(self.count):
            if self.givenAnswers[i] == self.trueAnswers[i]:
                if self.givenAnswers[i] == 1:
                    self.TP += 1
                else:
                    self.TN += 1
            else:
                if self.givenAnswers[i] == 1:
                    self.FP += 1
                else:
                    self.FN += 1
        self.TF = str(self.TP) + ' / ' + str(self.FP) + ' / ' + str(self.TN) + ' / ' + str(self.FN)

        """print('TP ', self.TP)
        print('FP ', self.FP)
        print('TN ', self.TN)
        print('FN ', self.FN)"""

    def set_answer_category(self):
        for i in range(self.count):
            if self.givenAnswers[i] == self.trueAnswers[i]:
                if self.givenAnswers[i] == 1:
                    self.TP += 1
                else:
                    self.TN += 1
            else:
                if self.givenAnswers[i] == 1:
                    self.FP += 1
                else:
                    self.FN += 1
        self.TF = str(self.TP) + ' / ' + str(self.FP) + ' / ' + str(self.TN) + ' / ' + str(self.FN)

        """print('TP ', self.TP)
        print('FP ', self.FP)
        print('TN ', self.TN)
        print('FN ', self.FN)"""

    def accuracy(self):
        return (self.TP + self.TN) / self.count

    def precision(self):
        if self.TP + self.FP > 0:
            return self.TP / (self.TP + self.FP)
        return 'Undefined'

    def recall(self):
        if self.TP + self.FN > 0:
            return self.TP / (self.TP + self.FN)
        return 'Undefined'

    def f1_score(self):
        recall = self.recall()
        precision = self.precision()

        if (recall != 'Undefined') and (precision != 'Undefined'):
            if recall + precision > 0:
                return (2 * recall * precision) / (recall + precision)
        return 'Undefined'

    def rmse(self):
        mse = np.sum((self.givenAnswers - self.trueAnswers) ** 2) / self.count
        return np.sqrt(mse)
