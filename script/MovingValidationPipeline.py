
# coding: utf-8

# In[ ]:

# Handle table-like data and matrices
import numpy as np
import pandas as pd

from collections import Counter

# Modelling Algorithms
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


# In[ ]:

from itertools import product
import gc


# In[ ]:

sales = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
items = pd.read_csv("../input/items.csv")
categories = pd.read_csv("../input/item_categories.csv")
print (sales.shape, test.shape)


# In[ ]:

# Add date_block_num for test with value 34 (next month)
test['date_block_num'] = 34


# In[ ]:

index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

#turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

#get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})
#fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
#join aggregated data to the grid
all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)

# Same as above but with shop-month aggregates
gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_shop':'sum'}})
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

# Same as above but with item-month aggregates
gb = sales.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_item':'sum'}})
gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

#sort the data
all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)


# In[ ]:

def downcast_dtypes(df):
    '''
        Changes column types in the dataframe:

                `float64` type to `float32`
                `int64`   type to `int32`
    '''

    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)

    return df


# In[ ]:

all_data = downcast_dtypes(all_data)

del grid, gb
gc.collect();


print("-------- All data loaded ------------")

# In[ ]:

all_data.head()


# In[ ]:

# List of columns that we will use to create lags
cols_to_rename = list(all_data.columns.difference(index_cols))


# In[ ]:

cols_to_rename


# In[ ]:

# lag months: 1 = last month sales; 12 = last year same month sales
shift_range = [1,2,3,4,5,12]


# In[ ]:

#import tqdm
from tqdm import tqdm


# In[ ]:

for month_shift in tqdm(shift_range):
    train_shift = all_data[index_cols + cols_to_rename].copy()

    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift



    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)

    # Merge lag feature with train data
    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

    # Merge lag feature with test data
    test = pd.merge(test, train_shift, on=index_cols, how='left').fillna(0)

del train_shift
gc.collect();


# In[ ]:

all_data.head()


# In[ ]:

# List of all lagged features
fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]
# We will drop these at fitting stage
to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num']

print fit_cols
print to_drop_cols


# ## Train / Validation split
print("-------- Moving window 5 fold validation ------------")
# In[ ]:

# Save `date_block_num`, as we can't use them as features, but will need them to split the dataset into parts
dates = all_data['date_block_num']

last_block = dates.max()
print('Test `date_block_num` is %d' % last_block)


# In[ ]:

lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'n_jobs': -1,
               'min_data_in_leaf': 2**7,
               'bagging_fraction': 0.75,
               'learning_rate': 0.03,
               'objective': 'mse',
               'bagging_seed': 2**7,
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0
              }

lgb = LGBMRegressor(**lgb_params)


# In[ ]:

# Moving window validation scheme.
# On each iteration, use last month for validation
validation_months = [33, 32, 31, 30, 29]

for last_month in validation_months:
    # Split train and validation data
    dates_train = dates[dates <  last_month]
    dates_test  = dates[dates == last_month]

    X_train = all_data.loc[dates <  last_month].drop(to_drop_cols, axis=1)
    X_test =  all_data.loc[dates == last_month].drop(to_drop_cols, axis=1)

    y_train = all_data.loc[dates <  last_month, 'target'].values
    y_test =  all_data.loc[dates == last_month, 'target'].values

    lgb.fit(X_train, y_train)

    pred_train = lgb.predict(X_train)
    pred_test = lgb.predict(X_test)

    ## R2 and RMSE score for each validation fold
    print('Month {0:d} Test R-2: {1:f}'.format(last_month, r2_score(y_test, pred_test)))
    print('Month {0:d} Test RMSE {1:f}'.format(last_month, np.sqrt(mean_squared_error(y_test, pred_test))))



# In[ ]:

del dates_train, dates_test, X_train, X_test, y_train, y_test, pred_train, pred_test
gc.collect()


print("-------- Prepare submission ------------")
# ## Prepare submission

# In[ ]:

## Take all train data
X_train_all = all_data.drop(to_drop_cols, axis=1)
y_train_all = all_data.target.values


# In[ ]:

X_train_all.target_lag_1.describe()


# In[ ]:

test.sort(columns='ID')


# In[ ]:

test_id = test.pop('ID')
test.drop('date_block_num', axis=1, inplace=True)


# In[ ]:

test.head()


# In[ ]:

X_train_all.head()


# In[ ]:

lgb.fit(X_train_all, y_train_all)

pred_train_all = lgb.predict(X_train_all)
pred_test = lgb.predict(test)

## R2 and RMSE score for each validation fold
print('Train R-2: {1:f}'.format(last_month, r2_score(y_train_all, pred_train_all)))
print('Train RMSE {1:f}'.format(last_month, np.sqrt(mean_squared_error(y_train_all, pred_train_all))))


# In[ ]:

pred_test.shape


# In[ ]:

for i in range(len(pred_test)):
    if pred_test[i] > 20:
        pred_test[i] = 20
    if pred_test[i] < 0:
        pred_test[i] = 20


# In[ ]:

test_submit = pd.DataFrame({'ID': test_id, 'item_cnt_month': pred_test})
print test_submit.shape
test_submit.to_csv('lgbm_all_1_2_3_4_5_12.csv', index=False)
test_submit.head()


# In[ ]:
