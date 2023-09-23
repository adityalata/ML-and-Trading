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
import matplotlib.pyplot as plt
import time


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903952381


def root_mean_squared_error(predictions, actual):
    return np.sqrt(((predictions - actual) ** 2).mean())


def r_squared(predictions, actual):
    corr_matrix = np.corrcoef(actual, predictions)
    return corr_matrix[0, 1]**2


if __name__ == "__main__":
    test_learner_start_time = time.process_time()
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
    verbose = False
  		  	   		  		 		  		  		    	 		 		   		 		  
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

    print('Starting Experiment 1')
    exp1_max_leaf_size = 30
    exp1_array_size = exp1_max_leaf_size+1
    exp1_rmse_in_sample = np.zeros(exp1_array_size)
    exp1_rmse_out_of_sample = np.zeros(exp1_array_size)
    exp1_xticks = []

    for i in range(exp1_array_size):
        exp1_trail_learner = dt.DTLearner(leaf_size=i, verbose=verbose)
        exp1_trail_learner.add_evidence(train_x, train_y)
        if i % 2:
            exp1_xticks.append(i)

        # in sample - train exp
        exp1_trail_pred_train_y = exp1_trail_learner.query(train_x)
        exp1_rmse_in_sample[i] = root_mean_squared_error(train_y, exp1_trail_pred_train_y)

        # out of sample - test exp
        exp1_trail_pred_test_y = exp1_trail_learner.query(test_x)
        exp1_rmse_out_of_sample[i] = root_mean_squared_error(exp1_trail_pred_test_y, test_y)

    exp1_max_rmse_in_sample = np.nanmax(exp1_rmse_in_sample)
    exp1_max_rmse_out_sample = np.nanmax(exp1_rmse_out_of_sample)
    exp1_min_rmse_out_sample = np.nanmin(exp1_rmse_out_of_sample)
    exp1_min_rmse_out_leaf_size = np.argsort(exp1_rmse_out_of_sample)[0]
    # print("exp1_rmse_in_sample ", exp1_rmse_in_sample)
    # print("exp1_rmse_out_of_sample ", exp1_rmse_out_of_sample)
    print('exp1_min_rmse_out_sample ', exp1_min_rmse_out_sample, " exp1_max_rmse_in_sample ", exp1_max_rmse_in_sample, " exp1_max_rmse_out_sample ", exp1_max_rmse_out_sample, "exp1_min_rmse_out_leaf_size", exp1_min_rmse_out_leaf_size)

    plt.figure(1)
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.axis([1, exp1_max_leaf_size, 0, max(exp1_max_rmse_in_sample, exp1_max_rmse_out_sample)])
    plt.axvline(x=exp1_min_rmse_out_leaf_size, color='r', label='Overfitting leaf size', linestyle='dashed')
    plt.xlabel('Leaf Size')
    plt.xticks(ticks=exp1_xticks, rotation=45)
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Figure 1: Overfitting trend in DT Learner - alata6')

    plt.plot(exp1_rmse_in_sample, label='In Sample Test')
    plt.plot(exp1_rmse_out_of_sample, label='Out of Sample Test')

    plt.legend(loc='lower right', shadow=True, fontsize='medium')
    plt.savefig('Experiment_1.png')

    print("====================================================================")

    print('Starting Experiment 2')
    exp2_max_leaf_size = 30
    exp2_bag_size = 20
    exp2_array_size = exp2_max_leaf_size + 1
    exp2_rmse_in_sample = np.zeros(exp2_array_size)
    exp2_rmse_out_of_sample = np.zeros(exp2_array_size)
    exp2_xticks = []

    for i in range(exp2_array_size):
        exp2_trail_learner = bl.BagLearner(bags=exp2_bag_size, verbose=verbose, kwargs={'leaf_size': i})
        exp2_trail_learner.add_evidence(train_x, train_y)
        if i % 2:
            exp2_xticks.append(i)

        # in sample - train exp
        exp2_trail_pred_train_y = exp2_trail_learner.query(train_x)
        exp2_rmse_in_sample[i] = root_mean_squared_error(train_y, exp2_trail_pred_train_y)

        # out of sample - test exp
        exp2_trail_pred_test_y = exp2_trail_learner.query(test_x)
        exp2_rmse_out_of_sample[i] = root_mean_squared_error(exp2_trail_pred_test_y, test_y)

    exp2_max_rmse_in_sample = np.nanmax(exp2_rmse_in_sample)
    exp2_max_rmse_out_sample = np.nanmax(exp2_rmse_out_of_sample)
    exp2_min_rmse_out_sample = np.nanmin(exp2_rmse_out_of_sample)
    exp2_min_rmse_out_leaf_size = np.argsort(exp2_rmse_out_of_sample)[0]
    # print("exp2_rmse_in_sample ", exp2_rmse_in_sample)
    # print("exp2_rmse_out_of_sample ", exp2_rmse_out_of_sample)
    print('exp2_min_rmse_out_sample ', exp2_min_rmse_out_sample, " exp2_max_rmse_in_sample ", exp2_max_rmse_in_sample,
          " exp2_max_rmse_out_sample ", exp2_max_rmse_out_sample, "exp2_min_rmse_out_leaf_size",
          exp2_min_rmse_out_leaf_size)

    plt.figure(2)
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.axis([1, exp2_max_leaf_size, 0, max(exp2_max_rmse_in_sample, exp2_max_rmse_out_sample, exp1_max_rmse_in_sample, exp1_max_rmse_out_sample)])
    plt.axvline(x=exp2_min_rmse_out_leaf_size, color='r', label='BagL - Overfitting leaf size', linestyle='dashed')
    plt.xlabel('Leaf Size')
    plt.xticks(ticks=exp2_xticks, rotation=45)
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Figure 2: Overfitting trend in Bag Learner - alata6')

    plt.plot(exp2_rmse_in_sample, label='BagL - In Sample Test')
    plt.plot(exp2_rmse_out_of_sample, label='BagL - Out of Sample Test')
    plt.plot(exp1_rmse_in_sample, label='DT - In Sample Test')
    plt.plot(exp1_rmse_out_of_sample, label='DT - Out of Sample Test')
    plt.axvline(x=exp1_min_rmse_out_leaf_size, color='b', label='DT - Overfitting leaf size', linestyle='dashed')

    plt.legend(loc='lower right', shadow=True, fontsize='medium')
    plt.savefig('Experiment_2.png')
    print("====================================================================")
    
    print('Starting Experiment 3')
    exp3_max_leaf_size = 30
    exp3_array_size = exp3_max_leaf_size+1
    exp3_dt_rmse_in_sample = np.zeros(exp3_array_size)
    exp3_dt_rmse_out_of_sample = np.zeros(exp3_array_size)
    exp3_rt_rmse_in_sample = np.zeros(exp3_array_size)
    exp3_rt_rmse_out_of_sample = np.zeros(exp3_array_size)
    exp3_dt_rsq_in_sample = np.zeros(exp3_array_size)
    exp3_dt_rsq_out_of_sample = np.zeros(exp3_array_size)
    exp3_rt_rsq_in_sample = np.zeros(exp3_array_size)
    exp3_rt_rsq_out_of_sample = np.zeros(exp3_array_size)
    exp3_dt_train_time = np.zeros(exp3_array_size)
    exp3_rt_train_time = np.zeros(exp3_array_size)
    exp3_xticks = []

    for i in range(exp3_array_size):
        exp3_dt_trail_learner = dt.DTLearner(leaf_size=i, verbose=verbose)
        exp3_trail_training_start_time = time.process_time()
        exp3_dt_trail_learner.add_evidence(train_x, train_y)
        exp3_dt_train_time[i] = time.process_time() - exp3_trail_training_start_time

        exp3_rt_trail_learner = rt.RTLearner(leaf_size=i, verbose=verbose)
        exp3_trail_training_start_time = time.process_time()
        exp3_rt_trail_learner.add_evidence(train_x, train_y)
        exp3_rt_train_time[i] = time.process_time() - exp3_trail_training_start_time
        if i % 2:
            exp3_xticks.append(i)

        # in sample - train exp
        exp3_dt_trail_pred_train_y = exp3_dt_trail_learner.query(train_x)
        exp3_dt_rmse_in_sample[i] = root_mean_squared_error(train_y, exp3_dt_trail_pred_train_y)
        exp3_dt_rsq_in_sample[i] = r_squared(train_y, exp3_dt_trail_pred_train_y)
        exp3_rt_trail_pred_train_y = exp3_rt_trail_learner.query(train_x)
        exp3_rt_rmse_in_sample[i] = root_mean_squared_error(train_y, exp3_rt_trail_pred_train_y)
        exp3_rt_rsq_in_sample[i] = r_squared(train_y, exp3_rt_trail_pred_train_y)

        # out of sample - test exp
        exp3_dt_trail_pred_test_y = exp3_dt_trail_learner.query(test_x)
        exp3_dt_rmse_out_of_sample[i] = root_mean_squared_error(exp3_dt_trail_pred_test_y, test_y)
        exp3_dt_rsq_out_of_sample[i] = r_squared(exp3_dt_trail_pred_test_y, test_y)
        exp3_rt_trail_pred_test_y = exp3_rt_trail_learner.query(test_x)
        exp3_rt_rmse_out_of_sample[i] = root_mean_squared_error(exp3_rt_trail_pred_test_y, test_y)
        exp3_rt_rsq_out_of_sample[i] = r_squared(exp3_rt_trail_pred_test_y, test_y)

    exp3_dt_max_rmse_in_sample = np.nanmax(exp3_dt_rmse_in_sample)
    exp3_dt_max_rmse_out_sample = np.nanmax(exp3_dt_rmse_out_of_sample)
    exp3_dt_max_rsq_in_sample = np.nanmax(exp3_dt_rsq_in_sample)
    exp3_dt_max_rsq_out_sample = np.nanmax(exp3_dt_rsq_out_of_sample)
    exp3_dt_min_rmse_out_sample = np.nanmin(exp3_dt_rmse_out_of_sample)
    exp3_dt_min_rmse_out_leaf_size = np.argsort(exp3_dt_rmse_out_of_sample)[0]
    exp3_rt_max_rmse_in_sample = np.nanmax(exp3_rt_rmse_in_sample)
    exp3_rt_max_rmse_out_sample = np.nanmax(exp3_rt_rmse_out_of_sample)
    exp3_rt_max_rsq_in_sample = np.nanmax(exp3_rt_rsq_in_sample)
    exp3_rt_max_rsq_out_sample = np.nanmax(exp3_rt_rsq_out_of_sample)
    # print("exp3_dt_rmse_in_sample ", exp3_dt_rmse_in_sample)
    # print("exp3_dt_rmse_out_of_sample ", exp3_dt_rmse_out_of_sample)
    print('exp3_dt_min_rmse_out_sample ', exp3_dt_min_rmse_out_sample, " exp3_dt_max_rmse_in_sample ", exp3_dt_max_rmse_in_sample, " exp3_dt_max_rmse_out_sample ", exp3_dt_max_rmse_out_sample, "exp3_dt_min_rmse_out_leaf_size", exp3_dt_min_rmse_out_leaf_size)

    plt.figure(3)
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.axis([1, exp3_max_leaf_size, 0, max(exp3_dt_max_rmse_in_sample, exp3_dt_max_rmse_out_sample, exp3_rt_max_rmse_in_sample, exp3_rt_max_rmse_out_sample)])
    plt.axvline(x=exp3_dt_min_rmse_out_leaf_size, color='r', label='DT Overfitting leaf size', linestyle='dashed')
    plt.xlabel('Leaf Size')
    plt.xticks(ticks=exp3_xticks, rotation=45)
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Figure 4: DT vs RT Learner - alata6')
    plt.plot(exp3_dt_rmse_in_sample, label='DT - In Sample Test')
    plt.plot(exp3_dt_rmse_out_of_sample, label='DT - Out of Sample Test')
    plt.plot(exp3_rt_rmse_in_sample, label='RT - In Sample Test')
    plt.plot(exp3_rt_rmse_out_of_sample, label='RT - Out of Sample Test')
    plt.legend(loc='lower right', shadow=True, fontsize='medium')
    plt.savefig('Experiment_3_rmse.png')

    plt.figure(4)
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.axis([1, exp3_max_leaf_size, 0, max(exp3_dt_max_rsq_in_sample, exp3_dt_max_rsq_out_sample, exp3_rt_max_rsq_in_sample,exp3_rt_max_rsq_out_sample)])
    plt.xlabel('Leaf Size')
    plt.xticks(ticks=exp3_xticks, rotation=45)
    plt.ylabel('Coefficient of Determination (R-Squared)')
    plt.title('Figure 3: DT vs RT Learner - alata6')
    plt.plot(exp3_dt_rsq_in_sample, label='DT - In Sample Test')
    plt.plot(exp3_dt_rsq_out_of_sample, label='DT - Out of Sample Test')
    plt.axvline(x=exp3_dt_min_rmse_out_leaf_size, color='r', label='DT Overfitting leaf size', linestyle='dashed')
    plt.plot(exp3_rt_rsq_in_sample, label='RT - In Sample Test')
    plt.plot(exp3_rt_rsq_out_of_sample, label='RT - Out of Sample Test')
    plt.legend(loc='lower right', shadow=True, fontsize='medium')
    plt.savefig('Experiment_3_rsq.png')

    plt.figure(5)
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.axis([1, exp3_max_leaf_size, 0, max(np.nanmax(exp3_dt_train_time), np.nanmax(exp3_rt_train_time))])
    plt.xlabel('Leaf Size')
    plt.xticks(ticks=exp3_xticks, rotation=45)
    plt.ylabel('Time to Train (Seconds)')
    plt.title('Figure 3: DT vs RT Learner - alata6')
    plt.plot(exp3_dt_train_time, label='DT - Time to Train')
    plt.plot(exp3_rt_train_time, label='RT - Time to Train')
    plt.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.savefig('Experiment_3_ttt.png')

    print("====================================================================")
    print("test learner execution completed in ", time.process_time()-test_learner_start_time, "seconds")
