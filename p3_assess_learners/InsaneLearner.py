import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = []
        for _ in range(20):  # InsaneLearner should contain 20 BagLearner instances where each instance is composed of 20 LinRegLearner instances
            self.learners.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, verbose=verbose))
    def author(self):
        return "alata6"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        each_learner_outputs = []
        for learner in self.learners:
            each_learner_outputs.append(learner.query(points))
        return np.mean(np.array(each_learner_outputs), axis=0)