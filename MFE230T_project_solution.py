#Parameters
min_training_size = 36
backtest_baseline = False
backtest_naive = False
backtest_quadrant = False
filter_features = True

#Import packages
from data_module import data_module
from backtest import run_backtest_from_positions
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix

#Set files and folders
data_folder = r'./Data/*.csv'

#Get pricing data, features, labels, and real labels
daily_df, features_df, labels_df, real_y = data_module(data_folder, 'M', min_training_size)

#Filter features
features_df = features_df.loc[:,['GSERCAUS','CESIUSD','GTII10','USGGBE10','ACMTP10','CPIAUCSL','GDP']].copy()

#Transform to numpy array (for scikit-learn)
X = np.array(features_df)
y = np.array(labels_df)

#Create Time Series splitter object with as many folds as n-1, so that it uses expanding window for training
tscv = TimeSeriesSplit(n_splits = features_df.shape[0]-1)

#Create dataframes to store predictions (to be used in backtester later)
#Baseline - Naive
predictions_df_naive = pd.read_csv(r'naive.csv', index_col = 0)
predictions_df_naive.index = features_df.ix[(min_training_size+1):,:].index
#Baseline - Quadrant
predictions_df_quadrant = pd.read_csv(r'quadrant.csv', index_col = 0)
#State-of-the-art machine learning model: simple ensemble (average) between Logistic Regression and Linear SVM
predictions_df = pd.DataFrame(index = labels_df.index, columns = labels_df.columns)
for train_index, test_index in tscv.split(X):
	if train_index.shape[0] < min_training_size:
		continue
	else:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		print('TRAIN:', train_index, 'TEST:', test_index)
		#Train model 1
		model1 = LogisticRegression()
		model2 = svm.LinearSVC(random_state = 123)
			
		for c in range(y_train.shape[1]):
			model1.fit(X_train, y_train[:,c])
			model2.fit(X_train, y_train[:,c])
			p1 = model1.predict(X_test)
			p2 = model2.predict(X_test)
			#Ensemble
			p = 0.5 * (p1 + p2)
			#Assign prediction to date and asset
			predictions_df.loc[predictions_df.index == labels_df.index[test_index][0], predictions_df.columns[c]] = p[0]

#Model evaluation
#Need to shift forward predictions as they are intended to be used on the next period's return. No need to do that with the quadrant baseline
predictions_df = predictions_df.shift()
predictions_df.dropna(inplace=True)

#Model had to have a starting number of data points to train so we need to chop the labels_df accordingly
real_y = real_y.loc[real_y.index.isin(predictions_df.index),:].copy()
predictions_df = predictions_df.astype(int)
real_y = real_y.astype(int)

##Backtest strategies
#Run backtests
backtest_results = run_backtest_from_positions(predictions_df, predictions_df_naive, predictions_df_quadrant)

#Show performance stats
print('/n')
backtest_results.display()
#Plot equity lines
print('/n')
backtest_results.plot(title = 'Equity Line Comparison', grid = True)