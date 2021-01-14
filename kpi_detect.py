import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys
import datetime
import time, re
import re
from joblib import Parallel, delayed
import sesd
from pyculiarity import detect_ts

import warnings
warnings.filterwarnings("ignore")
pd.set_option('mode.chained_assignment', None)


def fill_na_by_group_f(data):
    """利用所在同一小区同一小时的数据分组，每一组分别进行线性回归填充"""
    name = '上行流量(KByte)'
    x1 = data.groupby(['hour', '小区名称'])[name].apply(lambda group: group.interpolate(limit_direction='both'))
    data[name] = x1
    x2 = data.groupby('小区名称')[name].apply(lambda group: group.interpolate())
    data[name] = x2
    data[name] = data[name].fillna(method='ffill').fillna(method='bfill')
    # 填充CQI_ratio缺失值以及一开始便缺失的用户数
#     for name in ['上行流量(KByte)']:#, '下行流量(KByte)']:
#         data[name] = np.round(data[name].fillna(data.groupby(['hour', '小区名称'])[name].transform('mean')), 3)
    return data


def fill_na_by_group_b(data):
    """利用所在同一小区同一小时的数据分组，每一组分别进行线性回归填充"""
    name = '下行流量(KByte)'
    x1 = data.groupby(['hour', '小区名称'])[name].apply(lambda group: group.interpolate())
    data[name] = x1
    x2 = data.groupby('小区名称')[name].apply(lambda group: group.interpolate())
    data[name] = x2
    data[name] = data[name].fillna(method='ffill').fillna(method='bfill')
    # 填充CQI_ratio缺失值以及一开始便缺失的用户数
#     for name in ['上行流量(KByte)']:#, '下行流量(KByte)']:
#         data[name] = np.round(data[name].fillna(data.groupby(['hour', '小区名称'])[name].transform('mean')), 3)
    return data


def three_sigma_fw(x):
    '''
    df_col：DataFrame数据的某一列
    '''
    if x.mean_hour-2*x.std_hour > x['上行流量(KByte)']:
        return '流量突降'
    elif x.mean_hour+2*x.std_hour < x['上行流量(KByte)']:
        return '流量突升'
    else:
        return 0
    
    
def three_sigma_bw(x):
    '''
    df_col：DataFrame数据的某一列
    '''
    if x.mean_hour-2*x.std_hour>x['下行流量(KByte)']:
        return '流量突降'
    elif  x.mean_hour+2*x.std_hour<x['下行流量(KByte)']:
        return '流量突升'
    else:
        return 0


def apply_parallel(df_group, func):
    """利用 Parallel 和 delayed 函数实现并行运算"""
    results = Parallel(n_jobs=-1)(delayed(func)(group) for name, group in tqdm(df_group))
    return pd.concat(results)


def model_detect_pipeline_fw(df_concat_kpi, city, save_out=True):

	df_concat_kpi_fw = df_concat_kpi.pivot_table(index=["小区名称"], columns="开始时间", values="上行流量(KByte)")
	nan_num_f = (df_concat_kpi_fw.shape[1]-df_concat_kpi_fw.count(axis=1))
	cell_nan_100more_list_f = nan_num_f[nan_num_f >= 100]
	cell_nan_100less_list_f = nan_num_f[nan_num_f < 100]
	df_concat_kpi_f = df_concat_kpi[df_concat_kpi.小区名称.isin(cell_nan_100less_list_f.index)]
	df_concat_kpi_f = apply_parallel(df_concat_kpi_f.reset_index().groupby('小区名称'), fill_na_by_group_f)
	df_concat_kpi_fw = df_concat_kpi_f.pivot_table(index=["小区名称"], columns="开始时间", values="上行流量(KByte)")
	df_concat_kpi_fw_T = df_concat_kpi_fw.T
	cell_list_fw = df_concat_kpi_fw_T.columns

	cell_list_fw = cell_list_fw[:100]

	df_concat_kpi_fw_T = df_concat_kpi_fw_T.reset_index().rename(columns={'开始时间':'timestamp'})

	def detect_fw(cell):

	    try:
	        example_data = df_concat_kpi_fw_T[['timestamp',cell]]
	        example_data.loc[:,cell]=example_data[cell].fillna(method='ffill').fillna(method='bfill')
	        results = detect_ts(example_data, max_anoms=0.09, alpha=0.001, direction='both', only_last=None)
	        results['anoms']['cell']=cell
	        return results['anoms'].reset_index(drop=True)

	    except:
	        example_data = df_concat_kpi_fw_T[['timestamp',cell]]
	        example_data.loc[:,cell]=example_data[cell].fillna(method='ffill').fillna(method='bfill')
	        results = sesd.seasonal_esd(example_data[cell], periodicity=20, hybrid=True, max_anomalies=int(len(example_data[cell])*0.05))
	        tmp=example_data.loc[results]
	        tmp.columns=['timestamp','anoms']
	        tmp['cell']=cell

	        return tmp

	results_fw = Parallel(n_jobs=-1)(delayed(detect_fw)(name) for name in tqdm(cell_list_fw))
	model_label_fw = pd.concat(results_fw)
	# model_label_fw.to_csv('/home/share/ch/移动设计院/data/小区KPI/model_label_fw3.csv',index=False,encoding='utf_8_sig')

	df_concat_kpi_f_3 = df_concat_kpi_f[df_concat_kpi_f.开始时间 >(df_concat_kpi_f.开始时间.max()-datetime.timedelta(days=3))]
	df_concat_kpi_fw = df_concat_kpi_f_3.pivot_table(index=["小区名称"], columns="开始时间", values="上行流量(KByte)")
	df_concat_kpi_fw_T = df_concat_kpi_fw.T
	df_concat_kpi_fw_T = df_concat_kpi_fw_T.reset_index().rename(columns={'开始时间':'timestamp'})

	if 'index' in df_concat_kpi_fw_T.columns:
		df_concat_kpi_fw_T=df_concat_kpi_fw_T.drop('index', axis=1)
	df_concat_kpi_fw_T = df_concat_kpi_fw_T.set_index('timestamp')
	df_0label_kpi_fw_T = pd.DataFrame(list(df_concat_kpi_fw_T.index), columns=['timestamp'])
	for col in tqdm(df_concat_kpi_fw_T.columns):
		df_0label_kpi_fw_T[col] = df_concat_kpi_fw_T[col].rolling('5h').apply(lambda x: sum(x == 0)).values
	df_0label_kpi_fw_T = df_0label_kpi_fw_T.set_index('timestamp')
	df_concat_kpi_fw_T = df_concat_kpi_fw_T.reset_index()


	def select_zero(col):
	    tmp=df_0label_kpi_fw_T[col][ df_0label_kpi_fw_T[col]>1]
	    tmp_kpi=df_concat_kpi_fw_T[['timestamp',col]]
	    zero_list=[]
	    for idx in tmp.index:
	        tmp_kpi_idex=tmp_kpi[(tmp_kpi['timestamp']<=idx)&(tmp_kpi['timestamp']>=idx-datetime.timedelta(hours=5))]
	        tmp_kpi_idex=tmp_kpi_idex[tmp_kpi_idex[col]==0]
	        tmp_kpi_idex=tmp_kpi_idex.rename(columns={col:'value'})
	        tmp_kpi_idex['cell']=col
	        tmp_kpi_idex['label']='零流量'
	        zero_list.append(tmp_kpi_idex)
	    if len(zero_list)>0:
	        zero_df=pd.concat(zero_list)
	        return zero_df.drop_duplicates()
	    else:
	        return None
	results_fw_zero = Parallel(n_jobs=-1)(delayed(select_zero)(name) for name in tqdm(df_0label_kpi_fw_T.columns))
	results_fw_zero_df = pd.concat([re for re in results_fw_zero if re is not None])
	# results_fw_zero_df.to_csv('/home/share/ch/移动设计院/data/小区KPI/zero_label_fw3.csv',index=False,encoding='utf_8_sig')


	df_concat_kpi_f['weekday'] = df_concat_kpi_f.开始时间.apply(lambda x: x.weekday())
	df_concat_kpi_f['weekday'] = df_concat_kpi_f['weekday'].apply(lambda x:0 if x in [1,2,3,4,0] else 1)
	
	df_concat_kpi_f_3['weekday'] = df_concat_kpi_f_3.开始时间.apply(lambda x: x.weekday())
	df_concat_kpi_f_3['weekday'] = df_concat_kpi_f_3['weekday'].apply(lambda x:0 if x in [1,2,3,4,0] else 1)

	da_stat_hour_fw = df_concat_kpi_f.groupby(['小区名称','hour','weekday']).agg({"上行流量(KByte)": [np.mean, np.median, np.std]})
	da_stat_hour_fw.columns = ['mean_hour', 'median_hour', 'std_hour']
	da_stat_hour_fw.reset_index(inplace=True)

	df_concat_kpi_f_jump = pd.merge(df_concat_kpi_f_3[['小区名称', '所属地市',  '开始时间', '上行流量(KByte)', 'hour', 'weekday']],da_stat_hour_fw,on=['小区名称','hour','weekday'],how='left')
	df_concat_kpi_f_jump['fw_label'] = df_concat_kpi_f_jump.apply(lambda x: three_sigma_fw(x), axis=1)
	df_concat_kpi_f_jump.to_csv('/home/share/ch/移动设计院/data/小区KPI/上行流量_label_{}.csv'.format(city), index=False, encoding='utf_8_sig')

	# model_label_fw=pd.read_csv('/home/share/ch/移动设计院/data/小区KPI/model_label_fw3.csv')

	model_label_fw=model_label_fw.rename(columns={'timestamp':'开始时间','cell':'小区名称'})
	model_label_fw['lable_model']=1
	df_concat_kpi_f_jump=pd.merge(df_concat_kpi_f_jump[['小区名称','所属地市','开始时间','上行流量(KByte)','mean_hour','std_hour','fw_label']].astype(str),model_label_fw[['开始时间', '小区名称', 'lable_model']].astype(str),on=['开始时间', '小区名称'],how='left')

	# zero_label_fw=pd.read_csv("/home/share/ch/移动设计院/data/小区KPI/zero_label_fw3.csv")
	zero_label_fw=results_fw_zero_df
	zero_label_fw=zero_label_fw.rename(columns={'timestamp':'开始时间','cell':'小区名称','label':'label_zero'})

	df_concat_kpi_f_jump=pd.merge(df_concat_kpi_f_jump.astype(str),zero_label_fw[['开始时间', '小区名称', 'label_zero']].astype(str),on=['开始时间', '小区名称'],how='left')
	df_concat_kpi_f_jump.lable_model=df_concat_kpi_f_jump.lable_model.str.replace('nan','0')
	df_concat_kpi_f_jump.label_zero=df_concat_kpi_f_jump.label_zero.fillna('0')

	# df_concat_kpi_f_jump.to_csv('/home/share/ch/移动设计院/data/小区KPI/上行流量_label_{}.csv'.format(city),index=False,encoding='utf_8_sig')
	if save_out:
		table_zero=df_concat_kpi_f_jump[df_concat_kpi_f_jump.label_zero=='零流量'][['小区名称', '所属地市', '开始时间', '上行流量(KByte)','mean_hour', 'std_hour', 'label_zero']]
		table_zero=table_zero.rename(columns={'label_zero':'零流量','mean_hour':'同期历史均值','std_hour':'同期历史标准差'})
		table_lift=df_concat_kpi_f_jump[~(df_concat_kpi_f_jump.fw_label.isin(['0',0]))&~(df_concat_kpi_f_jump.lable_model.isin(['0',0]))][['小区名称', '所属地市', '开始时间', '上行流量(KByte)', 'mean_hour', 'std_hour', 'fw_label','lable_model']]
		table_lift=table_lift.rename(columns={'fw_label':'2-sigma准则','lable_model':'STL分解判断','mean_hour':'同期历史均值','std_hour':'同期历史标准差'})
		table_lift['STL分解判断']='异常'
		xlsx = pd.ExcelWriter('/home/share/ch/移动设计院/data/小区KPI/上行流量_异常_{}.xlsx'.format(city))
		table_lift.to_excel(xlsx, index=False, sheet_name='突升突降')
		table_zero.to_excel(xlsx, index=False, sheet_name='零流量')
		xlsx.close()
		return None
		df_concat_kpi_f_jump=df_concat_kpi_f_jump.rename(columns={'label_zero':'零流量','mean_hour':'同期历史均值','std_hour':'同期历史标准差','fw_label':'2-sigma准则','lable_model':'STL分解判断'})
		df_concat_kpi_f_jump.to_csv('/home/share/ch/移动设计院/data/小区KPI/上行流量_label_{}.csv'.format(city),index=False,encoding='utf_8_sig')
	else:
		return df_concat_kpi_f_jump


def model_detect_pipeline_bw(df_concat_kpi,city,save_out=True):
	df_concat_kpi_bw = df_concat_kpi.pivot_table(index=["小区名称"],columns="开始时间", values="下行流量(KByte)")
	nan_num_b=(df_concat_kpi_bw.shape[1]-df_concat_kpi_bw.count(axis=1))
	cell_nan_100more_list_b=nan_num_b[nan_num_b >= 100]
	cell_nan_100less_list_b=nan_num_b[nan_num_b < 100]
	df_concat_kpi_b=df_concat_kpi[df_concat_kpi.小区名称.isin(cell_nan_100less_list_b.index)]
	df_concat_kpi_b = apply_parallel(df_concat_kpi_b.reset_index().groupby('小区名称'), fill_na_by_group_b)
	df_concat_kpi_bw=df_concat_kpi_b.pivot_table(index=["小区名称"],columns="开始时间", values="下行流量(KByte)")
	df_concat_kpi_bw_T=df_concat_kpi_bw.T
	cell_list_bw=df_concat_kpi_bw_T.columns

	cell_list_bw = cell_list_bw[:100]


	df_concat_kpi_bw_T=df_concat_kpi_bw_T.reset_index().rename(columns={'开始时间':'timestamp'})

	def detect_bw(cell):
	    try:
	        example_data = df_concat_kpi_bw_T[['timestamp',cell]]
	        example_data.loc[:,cell]=example_data[cell].fillna(method='ffill').fillna(method='bfill')
	        results = detect_ts(example_data, max_anoms=0.09, alpha=0.001, direction='both', only_last=None)
	        results['anoms']['cell']=cell
	        return results['anoms'].reset_index(drop=True)
	    except:
	        example_data = df_concat_kpi_bw_T[['timestamp',cell]]
	        example_data.loc[:,cell]=example_data[cell].fillna(method='ffill').fillna(method='bfill')
	        results = sesd.seasonal_esd(example_data[cell], periodicity=20, hybrid=True, max_anomalies=int(len(example_data[cell])*0.05))
	        tmp=example_data.loc[results]
	        tmp.columns=['timestamp','anoms']
	        tmp['cell']=cell
	        return tmp

	results_bw = Parallel(n_jobs=-1)(delayed(detect_bw)(name) for name in tqdm(cell_list_bw))
	model_label_bw=pd.concat(results_bw)
	# model_label_bw.to_csv('/home/share/ch/移动设计院/data/小区KPI/model_label_bw3.csv',index=False,encoding='utf_8_sig')

	df_concat_kpi_b_3=df_concat_kpi_b[df_concat_kpi_b.开始时间>(df_concat_kpi_b.开始时间.max()-datetime.timedelta(days=3))]
	df_concat_kpi_bw=df_concat_kpi_b_3.pivot_table(index=["小区名称"],columns="开始时间", values="下行流量(KByte)")
	df_concat_kpi_bw_T=df_concat_kpi_bw.T
	df_concat_kpi_bw_T=df_concat_kpi_bw_T.reset_index().rename(columns={'开始时间':'timestamp'})

	if 'index' in df_concat_kpi_bw_T.columns:
		df_concat_kpi_bw_T=df_concat_kpi_bw_T.drop('index',axis=1)
	df_concat_kpi_bw_T=df_concat_kpi_bw_T.set_index('timestamp')
	df_0label_kpi_bw_T=pd.DataFrame(list(df_concat_kpi_bw_T.index),columns=['timestamp'])
	for col in tqdm(df_concat_kpi_bw_T.columns):
	    df_0label_kpi_bw_T[col]=df_concat_kpi_bw_T[col].rolling('5h').apply(lambda x:sum(x==0)).values

	df_concat_kpi_bw_T=df_concat_kpi_bw_T.reset_index()
	df_0label_kpi_bw_T=df_0label_kpi_bw_T.set_index('timestamp')

	def select_zero_bw(col):
	    tmp=df_0label_kpi_bw_T[col][ df_0label_kpi_bw_T[col]>1]
	    tmp_kpi=df_concat_kpi_bw_T[['timestamp',col]]
	    zero_list=[]
	    for idx in tmp.index:
	        tmp_kpi_idex=tmp_kpi[(tmp_kpi['timestamp']<=idx)&(tmp_kpi['timestamp']>=idx-datetime.timedelta(hours=5))]
	        tmp_kpi_idex=tmp_kpi_idex[tmp_kpi_idex[col]==0]
	        tmp_kpi_idex=tmp_kpi_idex.rename(columns={col:'value'})
	        tmp_kpi_idex['cell']=col
	        tmp_kpi_idex['label']='零流量'
	        zero_list.append(tmp_kpi_idex)
	    if len(zero_list)>0:
	        zero_df=pd.concat(zero_list)
	        return zero_df.drop_duplicates()
	    else:
	        return None
	results_bw_zero = Parallel(n_jobs=-1)(delayed(select_zero_bw)(name) for name in tqdm(df_0label_kpi_bw_T.columns))
	results_bw_zero_df=pd.concat([re for re in results_bw_zero if re is not None])
	# results_bw_zero_df.to_csv('/home/share/ch/移动设计院/data/小区KPI/zero_label_bw3.csv',index=False,encoding='utf_8_sig')

	df_concat_kpi_b['weekday'] = df_concat_kpi_b.开始时间.apply(lambda x: x.weekday())
	df_concat_kpi_b['weekday'] = df_concat_kpi_b['weekday'].apply(lambda x: 0 if x in [1, 2, 3, 4, 0] else 1)

	df_concat_kpi_b_3['weekday'] = df_concat_kpi_b_3.开始时间.apply(lambda x: x.weekday())
	df_concat_kpi_b_3['weekday'] = df_concat_kpi_b_3['weekday'].apply(lambda x: 0 if x in [1, 2, 3, 4, 0] else 1)

	da_stat_hour_bw=df_concat_kpi_b.groupby(['小区名称','hour','weekday']).agg({"下行流量(KByte)": [np.mean, np.median, np.std]})
	da_stat_hour_bw.columns = ['mean_hour', 'median_hour', 'std_hour']
	da_stat_hour_bw.reset_index(inplace=True)


	df_concat_kpi_b_jump=pd.merge(df_concat_kpi_b_3[['小区名称', '所属地市',  '开始时间', '下行流量(KByte)', 'hour', 'weekday']],da_stat_hour_bw,on=['小区名称','hour','weekday'],how='left')
	df_concat_kpi_b_jump['bw_label']=df_concat_kpi_b_jump.apply(lambda x: three_sigma_bw(x),axis=1)
	df_concat_kpi_b_jump.to_csv('/home/share/ch/移动设计院/data/小区KPI/下行流量_label_{}.csv'.format(city),index=False,encoding='utf_8_sig')

	# model_label_bw=pd.read_csv('/home/share/ch/移动设计院/data/小区KPI/model_label_bw3.csv')
	model_label_bw = model_label_bw.rename(columns={'timestamp':'开始时间', 'cell': '小区名称'})
	model_label_bw['lable_model'] = 1
	df_concat_kpi_b_jump=pd.merge(df_concat_kpi_b_jump[['小区名称','所属地市','开始时间','下行流量(KByte)','mean_hour','std_hour','bw_label']].astype(str),model_label_bw[['开始时间', '小区名称', 'lable_model']].astype(str),on=['开始时间', '小区名称'],how='left')

	# zero_label_bw=pd.read_csv("/home/share/ch/移动设计院/data/小区KPI/zero_label_bw3.csv")
	zero_label_bw=results_bw_zero_df

	zero_label_bw=zero_label_bw.rename(columns={'timestamp':'开始时间','cell':'小区名称','label':'label_zero'})

	df_concat_kpi_b_jump=pd.merge(df_concat_kpi_b_jump.astype(str),zero_label_bw[['开始时间', '小区名称', 'label_zero']].astype(str),on=['开始时间', '小区名称'],how='left')
	df_concat_kpi_b_jump.lable_model=df_concat_kpi_b_jump.lable_model.str.replace('nan','0')
	df_concat_kpi_b_jump.label_zero=df_concat_kpi_b_jump.label_zero.fillna('0')
	# df_concat_kpi_b_jump.to_csv('/home/share/ch/移动设计院/data/小区KPI/下行流量_label_{}.csv'.format(city),index=False,encoding='utf_8_sig')
	
	if save_out:
		table_zero=df_concat_kpi_b_jump[df_concat_kpi_b_jump.label_zero=='零流量'][['小区名称', '所属地市', '开始时间', '下行流量(KByte)','mean_hour', 'std_hour', 'label_zero']]
		table_zero=table_zero.rename(columns={'label_zero':'零流量','mean_hour':'同期历史均值','std_hour':'同期历史标准差'})
		table_lift=df_concat_kpi_b_jump[~(df_concat_kpi_b_jump.bw_label.isin(['0',0]))&~(df_concat_kpi_b_jump.lable_model.isin(['0',0]))][['小区名称', '所属地市', '开始时间', '下行流量(KByte)','mean_hour', 'std_hour', 'bw_label','lable_model']]
		table_lift=table_lift.rename(columns={'bw_label':'2-sigma准则','lable_model':'STL分解判断','mean_hour':'同期历史均值','std_hour':'同期历史标准差'})
		table_lift['STL分解判断']='异常'
		xlsx = pd.ExcelWriter('/home/share/ch/移动设计院/data/小区KPI/下行流量_异常_{}.xlsx'.format(city))
		table_lift.to_excel(xlsx, index=False, sheet_name='突升突降')
		table_zero.to_excel(xlsx, index=False, sheet_name='零流量')
		xlsx.close()
		return None
		df_concat_kpi_b_jump=df_concat_kpi_b_jump.rename(columns={'label_zero':'零流量','mean_hour':'同期历史均值','std_hour':'同期历史标准差','bw_label':'2-sigma准则','lable_model':'STL分解判断'})
		df_concat_kpi_b_jump.to_csv('/home/share/ch/移动设计院/data/小区KPI/下行流量_label_{}.csv'.format(city),index=False,encoding='utf_8_sig')
	else:
		return df_concat_kpi_b_jump



if __name__=='__main__':

	#需要检测的地市在列表内列出
	# for city in ['中山','湛江','江门','茂名','汕头','揭阳','梅州']:
	for city in ['茂名']:
		kpi_path = 'E:/dataset/广州智能运维数据/性能数据/{}/'.format(city)
		df_list = []
		for root, dirs, files in tqdm(os.walk(kpi_path)):
			for file in files:
				if file.endswith('.csv'):
					print(os.path.join(root, file))
					tmp_data = pd.read_csv(os.path.join(root, file),encoding='GBK')
					df_list.append(tmp_data)
		df_concat_kpi = pd.concat(df_list)
		del df_list
		# 小区数量较大的地区分批跑
		if city in ['广州', '深圳', '惠州', '东莞', '佛山', '中山', '湛江', '江门', '茂名', '汕头', '揭阳', '梅州']:
			cell_list = list(df_concat_kpi.小区名称.unique())
			n = len(cell_list)//12000
			kpi_f_jump_list = []
			kpi_b_jump_list = []
			for i in range(n+1):
				i = i+1
				if i == 1:
					tmp_cell_list = cell_list[:i*12000]
				elif i == n+1:
					tmp_cell_list = cell_list[(i-1)*12000:]
				else:
					tmp_cell_list = cell_list[(i-1)*12000:i*12000]
				cityt = city+str(i)
				print(cityt)
				df_concat_kpi_tmp = df_concat_kpi[df_concat_kpi.小区名称.isin(tmp_cell_list)]
				df_concat_kpi_tmp.开始时间 = pd.to_datetime(df_concat_kpi_tmp.开始时间)
				df_concat_kpi_tmp = df_concat_kpi_tmp.sort_values('开始时间')
				df_concat_kpi_tmp['hour'] = df_concat_kpi_tmp.开始时间.apply(lambda x: x.hour)
				df_concat_kpi_tmp = df_concat_kpi_tmp[~df_concat_kpi_tmp.hour.isin([23, 1, 3, 4, 5, 2, 6, 0])]
				df_concat_kpi_f_jump = model_detect_pipeline_fw(df_concat_kpi_tmp, cityt, save_out=False)
				df_concat_kpi_b_jump = model_detect_pipeline_bw(df_concat_kpi_tmp, cityt, save_out=False)
				kpi_f_jump_list.append(df_concat_kpi_f_jump)
				kpi_b_jump_list.append(df_concat_kpi_b_jump)
			kpi_f_jump = pd.concat(kpi_f_jump_list)
			del kpi_f_jump_list
			table_zero = kpi_f_jump[kpi_f_jump.label_zero == '零流量'][['小区名称', '所属地市', '开始时间', '上行流量(KByte)','mean_hour', 'std_hour', 'label_zero']]
			table_zero = table_zero.rename(columns={'label_zero':'零流量','mean_hour':'同期历史均值','std_hour':'同期历史标准差'})
			table_lift = kpi_f_jump[~(kpi_f_jump.fw_label.isin(['0',0]))&~(kpi_f_jump.lable_model.isin(['0',0]))][['小区名称', '所属地市', '开始时间', '上行流量(KByte)', 'mean_hour', 'std_hour', 'fw_label','lable_model']]
			table_lift = table_lift.rename(columns={'fw_label':'2-sigma准则','lable_model':'STL分解判断','mean_hour':'同期历史均值','std_hour':'同期历史标准差'})
			table_lift['STL分解判断']='异常'
			xlsx = pd.ExcelWriter('/home/share/ch/移动设计院/data/小区KPI/上行流量_异常_{}.xlsx'.format(city))
			table_lift.to_excel(xlsx, index=False, sheet_name='突升突降')
			table_zero.to_excel(xlsx, index=False, sheet_name='零流量')
			xlsx.close()
			kpi_f_jump=kpi_f_jump.rename(columns={'label_zero':'零流量','mean_hour':'同期历史均值','std_hour':'同期历史标准差','fw_label':'2-sigma准则','lable_model':'STL分解判断'})
			kpi_f_jump.to_csv('/home/share/ch/移动设计院/data/小区KPI/上行流量_label_{}.csv'.format(city),index=False,encoding='utf_8_sig')
			del kpi_f_jump
			kpi_b_jump=pd.concat(kpi_b_jump_list)
			del kpi_b_jump_list
			table_zero=kpi_b_jump[kpi_b_jump.label_zero=='零流量'][['小区名称', '所属地市', '开始时间', '下行流量(KByte)','mean_hour', 'std_hour', 'label_zero']]
			table_zero=table_zero.rename(columns={'label_zero':'零流量','mean_hour':'同期历史均值','std_hour':'同期历史标准差'})
			table_lift=kpi_b_jump[~(kpi_b_jump.bw_label.isin(['0',0]))&~(kpi_b_jump.lable_model.isin(['0',0]))][['小区名称', '所属地市', '开始时间', '下行流量(KByte)','mean_hour', 'std_hour', 'bw_label','lable_model']]
			table_lift=table_lift.rename(columns={'bw_label':'2-sigma准则','lable_model':'STL分解判断','mean_hour':'同期历史均值','std_hour':'同期历史标准差'})
			table_lift['STL分解判断']='异常'
			xlsx = pd.ExcelWriter('/home/share/ch/移动设计院/data/小区KPI/下行流量_异常_{}.xlsx'.format(city))
			table_lift.to_excel(xlsx, index=False, sheet_name='突升突降')
			table_zero.to_excel(xlsx, index=False, sheet_name='零流量')
			xlsx.close()
			kpi_b_jump=kpi_b_jump.rename(columns={'label_zero':'零流量','mean_hour':'同期历史均值','std_hour':'同期历史标准差','bw_label':'2-sigma准则','lable_model':'STL分解判断'})
			kpi_b_jump.to_csv('/home/share/ch/移动设计院/data/小区KPI/下行流量_label_{}.csv'.format(city),index=False,encoding='utf_8_sig')
			
			del kpi_b_jump

		else:
			cityt=city
			print(cityt)
			df_concat_kpi_tmp = df_concat_kpi
			df_concat_kpi_tmp.开始时间 = pd.to_datetime(df_concat_kpi_tmp.开始时间)
			df_concat_kpi_tmp = df_concat_kpi_tmp.sort_values('开始时间')
			df_concat_kpi_tmp['hour'] = df_concat_kpi_tmp.开始时间.apply(lambda x: x.hour)
			print(df_concat_kpi_tmp.shape)
			df_concat_kpi_tmp = df_concat_kpi_tmp[~df_concat_kpi_tmp.hour.isin([23, 1, 3, 4, 5, 2, 6, 0])]
			print(df_concat_kpi_tmp.shape)
			model_detect_pipeline_fw(df_concat_kpi_tmp, cityt)
			model_detect_pipeline_bw(df_concat_kpi_tmp, cityt)

