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
import numpy as np
import pandas as pd
from sympy import Matrix, GramSchmidt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
    return new_factors


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
    df2 = normalize(df, method='zscore')
    df3 = normalize(df, method='rank')
    # new_df = get_orthogonal_factors(arr, 'Schimidt')
    f6 = pd.DataFrame(np.random.randn(1000, 1), index=indices, columns=['f6'])
    new_f6 = orthogonal_factor(f6, df, 'Canonical')
    print('ok')
