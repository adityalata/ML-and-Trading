""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import math  		  	   		  		 		  		  		    	 		 		   		 		  
import sys  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903952381

if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		  		 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		  		 		  		  		    	 		 		   		 		  
        sys.exit(1)
    print("sys.argv[1] : ", sys.argv[1])
    if sys.argv[1].__contains__("Istanbul"):
        print("Cleaning Istanbul.csv")
        inf = open(sys.argv[1])
        rawdata = inf.readlines()
        rawRows = len(rawdata)
        rawCols = len(rawdata[0].strip().split(","))
        print("len rawdata", rawRows, " rawcols ", rawCols)
        data = np.zeros((rawRows-1, rawCols-1), dtype=float)
        print(" filteredData.shape ", data.shape)
        for rowCount in range(1, rawRows):
            columns = rawdata[rowCount].strip().split(",")
            # print("rowCount ", rowCount, " columns ", columns)
            data[rowCount-1, :] = columns[1:]
        print("filteredData ", data)
    else:
        inf = open(sys.argv[1])
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )
  		  	   		  		 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		  		 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows
    np.random.seed(gtid())
    # separate out training and testing data
    permutation = np.random.permutation(data.shape[0])
    col_permutation = np.random.permutation(data.shape[1] - 1)
    train_data = data[permutation[:train_rows], :]
    train_x = train_data[:, col_permutation]
    train_y = train_data[:, -1]
    test_data = data[permutation[train_rows:], :]
    test_x = test_data[:, col_permutation]
    test_y = test_data[:, -1]
  		  	   		  		 		  		  		    	 		 		   		 		  
    print("test_x.shape", f"{test_x.shape}")
    print("test_y.shape", f"{test_y.shape}")
    verbose = False  # todo make False before submission
  		  	   		  		 		  		  		    	 		 		   		 		  
    # create a learner and train it
    print("====================================================================")
    print("Lin Reg Learner")
    learner = lrl.LinRegLearner(verbose=verbose)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it  		  	   		  		 		  		  		    	 		 		   		 		  
    print(learner.author())  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		  		 		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("In sample results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		  		 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		  		 		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions  		  	   		  		 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  		 		  		  		    	 		 		   		 		  
    print()  		  	   		  		 		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		  		 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  	
    print("====================================================================")
    print('Decision Tree Learner')

    learner = dt.DTLearner(verbose=verbose)
    learner.add_evidence(train_x, train_y)
    print(learner.author())

    # evaluate in-sample
    pred_y = learner.query(train_x)
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])

    print()
    print('In-sample results')
    print('RMSE: {}'.format(rmse))

    c = np.corrcoef(pred_y, y=train_y)

    print('Corr: {}'.format(c[0,1]))

    # evaluate out-of-sample
    pred_y = learner.query(test_x)
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    print()
    print('Out-of-sample results')
    print('RMSE: {}'.format(rmse))

    c = np.corrcoef(pred_y, y=test_y)

    print('Corr: {}'.format(c[0, 1]))
    print("====================================================================")

    print('Random Tree Learner')

    learner = rt.RTLearner(verbose=verbose)
    learner.add_evidence(train_x, train_y)
    print(learner.author())

    # evaluate in-sample
    pred_y = learner.query(train_x)
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])

    print()
    print('In-sample results')
    print('RMSE: {}'.format(rmse))

    c = np.corrcoef(pred_y, y=train_y)

    print('Corr: {}'.format(c[0, 1]))

    # evaluate out-of-sample
    pred_y = learner.query(test_x)
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    print()
    print('Out-of-sample results')
    print('RMSE: {}'.format(rmse))

    c = np.corrcoef(pred_y, y=test_y)

    print('Corr: {}'.format(c[0, 1]))
    print("====================================================================")
    print('Bag Learner')

    learner = bl.BagLearner(verbose=verbose)  # defaults to 20 DTLearners with leaf size of 1
    learner.add_evidence(train_x, train_y)
    print(learner.author())

    # evaluate in-sample
    pred_y = learner.query(train_x)
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])

    print()
    print('In-sample results')
    print('RMSE: {}'.format(rmse))

    c = np.corrcoef(pred_y, y=train_y)

    print('Corr: {}'.format(c[0, 1]))

    # evaluate out-of-sample
    pred_y = learner.query(test_x)
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    print()
    print('Out-of-sample results')
    print('RMSE: {}'.format(rmse))

    c = np.corrcoef(pred_y, y=test_y)

    print('Corr: {}'.format(c[0, 1]))
    print("====================================================================")
    print('Insane Learner')

    learner = il.InsaneLearner(verbose=verbose)  # defaults to DTLearner with leaf size of 1
    learner.add_evidence(train_x, train_y)
    print(learner.author())

    # evaluate in-sample
    pred_y = learner.query(train_x)
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])

    print()
    print('In-sample results')
    print('RMSE: {}'.format(rmse))

    c = np.corrcoef(pred_y, y=train_y)

    print('Corr: {}'.format(c[0, 1]))

    # evaluate out-of-sample
    pred_y = learner.query(test_x)
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    print()
    print('Out-of-sample results')
    print('RMSE: {}'.format(rmse))

    c = np.corrcoef(pred_y, y=test_y)

    print('Corr: {}'.format(c[0, 1]))
    print("====================================================================")
