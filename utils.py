import numpy as np
import pandas as pd
import xlwings as xw

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR

from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ProgressBar
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.plots import plot_search_space
from sklearn_genetic.space import Integer, Continuous


from xgboost import XGBRegressor

def get_data(code):
    path = f'data/{code}.xls'
    app = xw.App(visible=False)
    book  = xw.Book(path)
    df = book.sheets(1).used_range.options(pd.DataFrame).value
    app.kill()
    df = df.sort_index(ascending=True)
    df = df.replace('', np.nan)
    df = df[['종가','BWI 20,2', 'ATR 14', 'Relative Volatility Index 14','Standard Deviation 10','Sigma 20','True Range']]
    df = df.loc[:, ~df.T.duplicated()]
    df = df.rename(columns={'종가':'Close'})
    df['return'] = np.log(df.Close) - np.log(df.Close.shift(1))
    df['std_1m'] = df.Close.rolling(20).std()
    df['std_2m'] = df.Close.rolling(40).std()
    df['std_3m'] = df.Close.rolling(60).std()
    df.columns = list(map(lambda x : code +'_'+x, df.columns.tolist()))
    df = df.dropna()
    df.index.name  = 'Date'
    return df

def calc_average_std(data):
    std_average = []
    for i in range(len(data)-20):
        temp = data.iloc[i:i+21]
        temp = temp / temp.iloc[0]
        temp['index_mean'] = temp.mean(axis=1)
        std_average.append(temp['index_mean'].std())
    return std_average


def weight_sampling(data, assets, trials=100000):    
    n_assets = len(assets)
    data = data[assets]
    daily_returns = np.log(data / data.shift(1))
    
    weights = []
    volatilities = []
    returns = []
    
    for _ in list(range(trials)):
        asset_weights = np.random.dirichlet(np.ones(n_assets))       
        weights.append(asset_weights)
        volatilities.append(np.sqrt(np.dot(asset_weights.T, np.dot(daily_returns.cov() * 252, asset_weights))))
        returns.append(np.sum(daily_returns.mean() * asset_weights) * 252)
        
    df = pd.DataFrame(weights, columns=assets)
    df['volatility'] = volatilities
    df['return'] = returns    
    
    return df


def portfolio_table(sampled_table):
    sampled_table['volatility_round'] = sampled_table['volatility'].round(2)
    sampled_table['neg_return'] = -sampled_table['return']
    sampled_table['negative_SR'] = sampled_table['neg_return'] / sampled_table['volatility'] 
    sampled_table = sampled_table.sort_values(by=['volatility_round', 'negative_SR'])
    sampled_table = sampled_table.drop_duplicates(subset=['volatility_round'])
    sampled_table = sampled_table.reset_index(drop=True)
    return sampled_table


def elasticnet_model():
    en_model = ElasticNet(max_iter=100000, random_state=42)

    en_param = {'l1_ratio': Continuous(0, 1),
                'alpha': Continuous(1e-6, 1e0, distribution='log-uniform'),
                'tol': Continuous(1e-6, 1e0, distribution='log-uniform')}

    en_clf = GASearchCV(estimator=en_model, param_grid=en_param, n_jobs=-1, 
                        verbose=False, scoring='r2', generations=300,
                        cv=TimeSeriesSplit())
    return en_clf
    
def randomforest_model():
    rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)

    rf_param = {'n_estimators': Integer(2, 200),
                'max_depth': Integer(2, 40),
                'min_samples_split': Integer(2, 10)}

    rf_clf = GASearchCV(estimator=rf_model, param_grid=rf_param, n_jobs=-1, verbose=False, scoring='r2', generations=60,
                        cv=TimeSeriesSplit())
    
    return rn_clf


def xgboost_model():
    xgb_model = XGBRegressor(n_jobs=-1, random_state=42)
    xgb_param = {'n_estimators': Integer(2, 600),
                 'max_depth': Integer(1, 40),
                 'reg_alpha': Continuous(1e-6, 1e2, distribution='log-uniform'),
                 'reg_lambda': Continuous(1e-6, 1e2, distribution='log-uniform')}

    xgb_clf = GASearchCV(estimator=xgb_model, param_grid=xgb_param, n_jobs=-1, verbose=False, scoring='r2', generations=60,
                         cv=TimeSeriesSplit())
    return xgb_clf

def svr_model():
    svr_model = SVR()
    svr_param = {'C': Continuous(1e-3, 1e8, distribution='log-uniform'),
                 'gamma': Continuous(1e-6, 1e5, distribution='log-uniform')}
    svr_clf = GASearchCV(estimator=svr_model, param_grid=svr_param, n_jobs=-1, verbose=False, scoring='r2', generations=300,
                         cv=TimeSeriesSplit())
    return svr_clf