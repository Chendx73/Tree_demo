import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression


def load_dataset():
    print("载入数据...")
    train_df = pd.read_csv('Ad_click_prediction_train.csv', sep=',',usecols=['session_id','user_id','product'])
    test_df = pd.read_csv('Ad_Click_prediciton_test.csv', sep=',')
    print("训练集的大小：", len(train_df))
    print("测试集的大小：", len(test_df))
    return train_df, test_df

def preprocess(data_df):
    # data_df['DateTime']=
    pass

def process_dataset(train_df, test_df):
    print("处理数据...")
    y_train = train_df.iloc[:, -1]
    y_test = test_df.iloc[:, -1]
    X_train = train_df.iloc[:, :-1]
    X_test = test_df.iloc[:, :-1]
    lgb_trn = lgb.Dataset(X_train, y_train)
    lgb_tst = lgb.Dataset(X_test, y_test)
    return lgb_trn, lgb_tst


def train(lgb_trn):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 63,
        'num_trees': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('开始训练...')
    gbm = lgb.train(params, lgb_trn, num_boost_round=100, valid_sets=lgb_trn)
    print('保存模型...')
    gbm.save_model('model.txt')


def predict(X_train):
    num_leaf = 63
    print('读取模型...')
    model = lgb.Booster(model_file='model.txt')
    print('开始预测...')
    y_pred = model.predict(X_train, pred_leaf=True)
    transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
        transformed_training_matrix[i][temp] += 1


if __name__ == '__main__':
    train_df, test_df = load_dataset()
    lgb_trn, lgb_tst = process_dataset(train_df, test_df)
    train(lgb_trn)
    predict(lgb_trn.data)
