import numpy as np
import DTLearner as dt

class BagLearner(object):
    """
    This is a regression Bag Learner (i.e., a BagLearner containing Random Trees).

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner=dt.DTLearner, kwargs={'leaf_size': 1}, bags=20, boost=False, verbose=False):
        """
        Constructor method
        """
        self.verbose = verbose
        learners = []
        for i in range(0, bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.boost = boost
        self.bag_count = bags
        np.random.seed(self.gtid())  # todo check

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "alata6"  # replace tb34 with your Georgia Tech username

    def gtid(self):
        """
        :return: The GT ID of the student
        :rtype: int
        """
        return 903952381  # replace with your GT ID number

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        data = np.column_stack((data_x, data_y))   # Stack 1-D arrays as columns into a 2-D array.
        number_elements = data.shape[0]  # amount of training data

        for learner in self.learners:  # todo handle case where learner is not defined
            bag = np.empty(shape=(0, data.shape[1]))

            for _ in range(number_elements):  # todo verify if floor * 0.6 req
                index = np.random.randint(0, number_elements)  # sample with replacement
                bag = np.row_stack((bag, data[index]))  # Stack arrays in sequence vertically (row wise).

            learner.add_evidence(bag[:, 0:-1], bag[:, -1])  # training each learner

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        each_learner_outputs = []
        for learner in self.learners:
            each_learner_outputs.append(learner.query(points))
        return np.mean(each_learner_outputs, axis=0)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
