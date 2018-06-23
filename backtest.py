###IMPORT PACKAGES###
import bt
import pandas as pd

def run_backtest_from_positions(predictions_df1, predictions_df2, predictions_df3):
	data = pd.read_excel(r'data.xlsx', index_col = 0)	
	###ML SOLUTION###
	#Expand positions_df to daily frequency
	positions_df1 = pd.DataFrame(index = data.index)
	positions_df1 = pd.merge(positions_df1, predictions_df1, how = 'left', left_index = True, right_index = True)
	positions_df1.fillna(method = 'ffill', inplace = True)
	positions_df1.dropna(inplace=True)
	positions_df1 = positions_df1.divide(((positions_df1.abs()).sum(axis=1)), axis = 'index')
	#Match starting date on data_df
	data1 = data.loc[data.index.isin(positions_df1.index),:].copy()
	#Create strategy object
	s1 = bt.Strategy('ML Solution', [bt.algos.WeighTarget(positions_df1), bt.algos.Rebalance()])
	t1 = bt.Backtest(s1, data1)

	###BASELINE NAIVE###
	positions_df2 = pd.DataFrame(index = data.index)
	positions_df2 = pd.merge(positions_df2, predictions_df2, how = 'left', left_index = True, right_index = True)
	positions_df2.fillna(method = 'ffill', inplace = True)
	positions_df2.dropna(inplace=True)
	positions_df2 = positions_df2.divide(((positions_df2.abs()).sum(axis=1)), axis = 'index')
	#Match starting date on data_df
	data2 = data.loc[data.index.isin(positions_df2.index),:].copy()
	#Create strategy object
	s2 = bt.Strategy('Baseline - Naive', [bt.algos.WeighTarget(positions_df2), bt.algos.Rebalance()])
	t2 = bt.Backtest(s2, data2)

	###BASELINE QUADRANT###
	positions_df3 = pd.DataFrame(index = data.index)
	positions_df3 = pd.merge(positions_df3, predictions_df3, how = 'left', left_index = True, right_index = True)
	positions_df3.fillna(method = 'ffill', inplace = True)
	positions_df3.dropna(inplace=True)
	positions_df3 = positions_df3.divide(((positions_df3.abs()).sum(axis=1)), axis = 'index')
	#Match starting date on data_df
	data3 = data.loc[data.index.isin(positions_df3.index),:].copy()
	#Create strategy object
	s3 = bt.Strategy('Baseline - Quadrant', [bt.algos.WeighTarget(positions_df3), bt.algos.Rebalance()])
	t3 = bt.Backtest(s3, data3)

	#Run backtest
	res = bt.run(t1, t2, t3)

	return res