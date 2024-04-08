#! /usr/bin/env python
# -*- coding: utf-8-*-
"""
data preprocess utils for factor analysis

factors : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the values for factors.
        ::
            ---------------------------------------
                       |       | f1  | f2  | f3
            ---------------------------------------
                date   | asset |     |     |
            ---------------------------------------
                       | AAPL  | 0.09|-0.01|-0.079
                       ----------------------------
                       | BA    | 0.02| 0.06| 0.020
                       ----------------------------
            2014-01-01 | CMG   | 0.03| 0.09| 0.036
                       ----------------------------
                       | DAL   |-0.02|-0.06|-0.029
                       ----------------------------
                       | LULU  |-0.03| 0.05|-0.009
                       ----------------------------

@author:
@date: 2023/9/20
"""
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed

from tqdm import tqdm
import numpy as np
import pandas as pd
from sympy import Matrix, GramSchmidt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm


# na/nan values process
def process_na(factors, method='fill', **kwargs):
    if method == 'drop':
        res = factors.dropna()
    elif method == 'fill':
        value = kwargs.get('value', 0)
        if isinstance(value, (int, float)):
            res = factors.fillna(value=value)
        elif value in ['ffill', 'bfill']:   # NOTE: 'bfill'可能引入穿越问题
            res = factors.groupby(level=1).fillna(method=value)
        elif value == 'mean':
            res = factors.groupby(level=0).apply(lambda df: df.fillna(value=df.mean()))
        elif value == 'median':
            res = factors.groupby(level=0).apply(lambda df: df.fillna(value=df.median()))
    else:
        return factors
    return res


# outliers/inf process
def _outliers_std(df, n):
    mean_ = df.mean()
    std_ = df.std()+0.00001
    deviation = (df-mean_)/std_
    df = df.where(deviation.abs()<n, mean_ + np.sign(deviation)*n*std_)
    return df

def _outliers_mad(df, n):
    mean_ = df.mean()
    mad_ = (df - mean_).abs().mean() + 0.00001
    deviation = (df-mean_)/mad_
    df = df.where(deviation.abs()<n, mean_ + np.sign(deviation)*n*mad_)
    return df

def _outliers_percentile(df, percentiles):
    df_percents = df.quantile(percentiles)
    df = df.where(df>df_percents.loc[percentiles[0]], df_percents.loc[percentiles[0]], axis=1)
    df = df.where(df<df_percents.loc[percentiles[1]], df_percents.loc[percentiles[1]], axis=1)
    return df

def process_outliers(factors, method='std', **kwargs):
    if method == 'std':
        res = factors.groupby(level=0).apply(_outliers_std, n=kwargs.get('n', 3))
    elif method == 'mad':
        res = factors.groupby(level=0).apply(_outliers_mad, n=kwargs.get('n', 3))
    elif method == 'percentile':
        percentiles = kwargs.get('percentile', (0.05, 0.95))
        if isinstance(percentiles, (int, float)):
            if percentiles >= 1:
                percentiles = percentiles / 100
            percentiles = sorted((1 - percentiles, percentiles))
        res = factors.groupby(level=0).apply(_outliers_percentile, percentiles=percentiles)
    else:
        raise ValueError(f'Unsupported method: {method}')
    res = res.droplevel(level=0)
    return res


# normalization
def normalize(factors, method='zscore', **kwargs):
    if method == 'minmax':
        res = factors.groupby(level=0).apply(lambda x: pd.DataFrame(MinMaxScaler().fit_transform(x.values),
                                                                    index=x.index, columns=x.columns))
    elif method == 'zscore':
        res = factors.groupby(level=0).apply(lambda x: pd.DataFrame(StandardScaler().fit_transform(x.values),
                                                                    index=x.index, columns=x.columns))
    elif method == 'rank':
        res = factors.groupby(level=0).apply(lambda x:x.rank(method='average', pct=True))
    else:
        raise ValueError(f'Unsupported method: {method}')
    res = res.droplevel(level=0)
    return res


# neutralization


# orthogonal
def _single_orthogonal_factors(factors, method='Schimidt'):
    """
    factors: 2D np array, N*K
    method: str, Schimidt/Canonical/Symmetry
    """
    if method == 'Schimidt':
        arr = [Matrix(x) for x in factors.T]
        new_factors = GramSchmidt(arr, orthonormal=False)
        new_factors = np.array(new_factors)[:, :, 0].T
    else:
        # 矩阵算法仅适用于标准化后的factors
        M = (factors.shape[0] - 1) * np.cov(factors.T)
        eigen_values, U = np.linalg.eig(M)
        D = np.diag(eigen_values ** (-0.5))
        if method == 'Canonical':
            S = U @ D
            new_factors = factors @ S
        elif method == 'Symmetry':
            S = U @ D @ U.T
            new_factors = factors @ S
        else:
            raise ValueError(f'unsupported method {method}, should be one of Schimidt/Canonical/Symmetry')

    return new_factors


def get_orthogonal_factors(factors, method='Schimidt'):
    """
    get orthogonal factors with each factor is orthogonal to other factors

    Parameters
    ----------
    factors : pd.DataFrame - MultiIndex

    method : str
        Orthogonal method: 'Schimidt', 'Symmetry', 'Canonical'

    Returns
    -------
    orthogonal_factors: pd.DataFrame - MultiIndex
    """
    dates = factors.index.get_level_values(level='date').unique()
    factor_names = factors.columns
    new_factors = []
    for dt in dates:
        tmp_factor = factors.loc[dt]
        tmp_new = _single_orthogonal_factors(tmp_factor.values, method=method)
        tmp_new = pd.DataFrame(tmp_new, index=tmp_factor.index, columns=factor_names)
        tmp_new['date'] = dt
        tmp_new.set_index(['date', tmp_new.index], inplace=True)
        new_factors.append(tmp_new)
    new_factors = pd.concat(new_factors)
    new_factors.sort_values(['date', 'asset'], inplace=True)
    return new_factors


def _process(tmp_factor, name, method):
    factor_names = tmp_factor.columns
    tmp_new = _single_orthogonal_factors(tmp_factor.values, method=method)
    tmp_new = pd.DataFrame(tmp_new, index=tmp_factor.index, columns=factor_names)
    # tmp_new['date'] = name
    # tmp_new.set_index(['date', tmp_new.index], inplace=True)
    return tmp_new


def get_orthogonal_factors_mp(factors, method='Schimidt'):
    """
    get orthogonal factors with each factor is orthogonal to other factors

    Parameters
    ----------
    factors : pd.DataFrame - MultiIndex

    method : str
        Orthogonal method: 'Schimidt', 'Symmetry', 'Canonical'

    Returns
    -------
    orthogonal_factors: pd.DataFrame - MultiIndex
    """
    groups = factors.groupby(level='date')
    new_factors = []
    total_grps = len(groups)
    finished = 0
    with tqdm(total=total_grps) as pbar:
        with ProcessPoolExecutor(max_workers=4) as pool:
            results = {pool.submit(_process, grp, name, method): name for name, grp in groups}

            for res in as_completed(results):
                tmp_res = res.result()
                new_factors.append(tmp_res)
                finished += 1
                pbar.update(1)

    new_factors = pd.concat(new_factors)
    new_factors.sort_values(['date', 'asset'], inplace=True)
    return new_factors


def get_residual(factor, base_factors, dt=None):
    """" 单期，线性回归残差 """
    y = factor.values
    X = base_factors.values
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    res = model.resid
    res = pd.DataFrame(res, index=factor.index, columns=factor.columns)
    if dt is not None:
        res['date'] = dt
        res = res.set_index('date', append=True).swaplevel()
    return res


def residual_factor(factor, base_factors):
    dates = factor.index.unique(level='date')
    res = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = []
        for dt in dates:
            tmp_factor = factor.loc[dt].fillna(value=0)
            tmp_base = base_factors.loc[dt].reindex(tmp_factor.index).fillna(value=0)
            future = pool.submit(get_residual, tmp_factor, tmp_base, dt)
            jobs.append(future)

        for future in as_completed(jobs):
            res.append(future.result())

    res = pd.concat(res)
    res.sort_index(level=['date', 'code'], inplace=True)
    return res


def orthogonal_factor(factor, base_factors, method='Schimidt'):
    """ get orthogonal factor of factor to base_factors
    """
    # make sure factor in last column
    factors = pd.merge(base_factors, factor, left_index=True, right_index=True, how='outer')
    orth_factors = get_orthogonal_factors(factors, method=method)
    new_factor = orth_factors.iloc[:, [-1]]
    return new_factor


# def Schimidt(factors):
#     """ 循环算法，带归一化 """
#     col_name = factors.columns
#     index = factors.index
#     factors = factors.values
#
#     R = np.zeros((factors.shape[1], factors.shape[1]))
#     Q = np.zeros(factors.shape)
#     for k in range(0, factors.shape[1]):
#         R[k, k] = np.sqrt(np.dot(factors[:, k], factors[:, k]))
#         Q[:, k] = factors[:, k] / R[k, k]
#         for j in range(k + 1, factors.shape[1]):
#             R[k, j] = np.dot(Q[:, k], factors[:, j])
#             factors[:, j] = factors[:, j] - R[k, j] * Q[:, k]
#
#     Q = pd.DataFrame(Q, columns=col_name, index=index)
#     return Q
#


if __name__ == '__main__':
    indices = [['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'], range(200)]
    indices = pd.MultiIndex.from_product(iterables=indices, names=['date', 'asset'])
    columns = ['f1', 'f2', 'f3', 'f4', 'f5']
    df = pd.DataFrame(np.random.randn(1000, 5), index=indices, columns=columns)
    # df_1 = process_outliers(df, method='std')
    # df_2 = process_outliers(df, method='mad')
    # df_3 = process_outliers(df, method='percentile')
    df1 = normalize(df, method='minmax')
    # df2 = normalize(df, method='zscore')
    # df3 = normalize(df, method='rank')
    f6 = pd.DataFrame(np.random.randn(1000, 1), index=indices, columns=['f6'])
    df_res = get_residual(f6, df)
    new_df = get_orthogonal_factors(df1, 'Schimidt')
    print(new_df.head())
    new_df2 = get_orthogonal_factors_mp(df1, 'Schimidt')
    print(new_df2.head())
    print(np.testing.assert_almost_equal(new_df.values, new_df2.values))
    # new_f6 = orthogonal_factor(f6, df, 'Canonical')
    print('ok')
