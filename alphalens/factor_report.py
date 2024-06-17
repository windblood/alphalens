#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
desc: 单因子因子分析、及正交化后因子分析的对比
TODO: 输入多个因子，按单因子循环，loader需要可重用
0. 基础因子从config.py读取base_factors或style_factors；正则化默认回归残差
1. 边际效果变化
2. 原因子分析报告
3. 正交化后因子报告

@author: admin
@date: 2024/3/26
"""
from datetime import datetime, date
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import empyrical as ep

import config
from loader import MysqlLoader, CSVLoader
from analyzer.classic_analyzer import ClassicAnalyzer
from analyzer.topK_analyzer import TopKAnalyzer
from preprocess import orthogonal_factor, get_residual
from backtest import Backtest

loaders = {'mysql': MysqlLoader, 'csv': CSVLoader}
analyzers = {'classic': ClassicAnalyzer, 'topk': TopKAnalyzer}


def alpha_report(cfg):
    if 'factor_name' not in cfg:
        raise KeyError()
    if 'start_date' not in cfg:
        raise KeyError()

    factor_name = cfg['factor_name']
    start_date = cfg['start_date']
    end_date = cfg.get('end_date', date.today())

    loader = cfg.get('loader', 'mysql')
    loader_cls = loaders.get(loader, MysqlLoader)

    analyzer = cfg.get('analyzer', 'classic')
    analyzer_cls = analyzers.get(analyzer, ClassicAnalyzer)

    kwargs = cfg.copy()
    dataloader = loader_cls(**kwargs)

    bench_factors = cfg.get('bench_factors', config.style_factors)
    periods = cfg.get('periods', (1, 5, 10))
    method = cfg.get('orthogonal_method', 'Residual')

    groupby = cfg.get('groupby', False)
    if groupby:
        groupby = dataloader.get_stock_industry(start_dt=start_date, end_dt=end_date)
    else:
        groupby = None

    weights = cfg.get('weights', False)
    if weights:
        weights = dataloader.get_industry_weight(start_dt=start_date, end_dt=end_date)
    else:
        weights = 1.0

    target_factor = dataloader.get_factor_data([factor_name], start_dt=start_date, end_dt=end_date)
    bench_factors = dataloader.get_factor_data(bench_factors, start_dt=start_date, end_dt=end_date)
    prices = dataloader.get_stock_price(start_dt=start_date, end_dt=end_date)
    prices['date'] = pd.to_datetime(prices['date'].astype(str))
    prices = pd.pivot_table(prices, index="date", columns="code", values='close')
    # groupby fill na
    target_factor = target_factor.groupby(level='code').ffill()
    bench_factors = bench_factors.groupby(level='code').ffill()
    prices.ffill(inplace=True)

    if method == 'Residual':
        marginal_factor = get_residual(target_factor, bench_factors)
    else:
        marginal_factor = orthogonal_factor(target_factor, bench_factors, method=method)
    raw_analyzer = analyzer_cls(factor=target_factor[factor_name], prices=prices, periods=periods,
                                groupby=groupby, weights=weights, **kwargs)
    marginal_analyzer = analyzer_cls(factor=marginal_factor[factor_name], prices=prices, periods=periods,
                                     groupby=groupby, weights=weights, **kwargs)

    # TODO: 进程并行
    raw_analyzer.create_report(factor_name)
    marginal_analyzer.create_report(f'{factor_name}_orth')

    # 对比报告：收益率、IC
    raw_returns = raw_analyzer.plot_returns_table(make_pretty=False)
    marginal_returns = marginal_analyzer.plot_returns_table(make_pretty=False)
    returns = pd.merge(raw_returns, marginal_returns, left_index=True, right_index=True,
                       suffixes=('_raw', '_orth'))
    raw_ic = raw_analyzer.plot_information_table(make_pretty=False)
    marginal_ic = marginal_analyzer.plot_information_table(make_pretty=False)
    ics = pd.merge(raw_ic, marginal_ic, left_index=True, right_index=True,
                       suffixes=('_raw', '_orth'))
    compare_df = pd.concat([returns, ics], axis=0)
    compare_df.to_csv(f'compares_{factor_name}.csv', index=True)
    return compare_df


def backtest_report(cfg):
    if 'factor_names' not in cfg:
        raise KeyError()
    if 'start_date' not in cfg:
        raise KeyError()

    factor_name = cfg['factor_name']
    start_date = cfg['start_date']
    end_date = cfg.get('end_date', date.today())

    loader = cfg.get('loader', 'mysql')
    loader_cls = loaders.get(loader, MysqlLoader)

    kwargs = cfg.copy()
    dataloader = loader_cls(**kwargs)

    bench_factors = cfg.get('bench_factors', config.style_factors)

    groupby = cfg.get('groupby', False)
    if groupby:
        groupby = dataloader.get_stock_industry(start_dt=start_date, end_dt=end_date)
    else:
        groupby = None

    weights = cfg.get('weights', False)
    if weights:
        weights = dataloader.get_industry_weight(start_dt=start_date, end_dt=end_date)
    else:
        weights = 1.0

    target_factor = dataloader.get_factor_data(factor_name, start_date, end_date)
    bench_factors = dataloader.get_factor_data(bench_factors, start_date, end_date)
    prices = dataloader.get_stock_price(start_dt=start_date, end_dt=end_date)

    # TODO: 进程并行, Backtest生成放到函数里？
    bench_bt = Backtest(factors=bench_factors, prices=prices, group=groupby, group_weight=weights,
                        factor_weight=None, initial_asset=10000000)
    factor_bt = Backtest(factors=bench_factors.merge(target_factor, left_index=True, right_index=True),
                         prices=prices, group=groupby, group_weight=weights, factor_weight=None,
                         initial_asset=10000000)


def list_factors():
    return []


def load_factor(factor_name, start_date, end_date):
    return pd.DataFrame()


def alphas_report(cfg):
    if 'start_date' not in cfg:
        raise KeyError()

    start_date = cfg['start_date']
    end_date = cfg.get('end_date', date.today())

    loader = cfg.get('loader', 'mysql')
    loader_cls = loaders.get(loader, MysqlLoader)

    analyzer = cfg.get('analyzer', 'classic')
    analyzer_cls = analyzers.get(analyzer, ClassicAnalyzer)

    kwargs = cfg.copy()
    dataloader = loader_cls(**kwargs)

    bench_factors = cfg.get('bench_factors', config.style_factors)
    periods = cfg.get('periods', (1, 5, 10))
    method = cfg.get('orthogonal_method', 'Residual')

    groupby = cfg.get('groupby', False)
    if groupby:
        groupby = dataloader.get_stock_industry(start_dt=start_date, end_dt=end_date)
    else:
        groupby = None

    weights = cfg.get('weights', False)
    if weights:
        weights = dataloader.get_industry_weight(start_dt=start_date, end_dt=end_date)
    else:
        weights = 1.0


    bench_factors = dataloader.get_factor_data(bench_factors, start_dt=start_date, end_dt=end_date)
    prices = dataloader.get_stock_price(start_dt=start_date, end_dt=end_date)
    prices['date'] = pd.to_datetime(prices['date'].astype(str))
    prices = pd.pivot_table(prices, index="date", columns="code", values='close')
    # groupby fill na

    bench_factors = bench_factors.groupby(level='code').ffill()
    prices.ffill(inplace=True)

    factors = list_factors()
    for factor_name in tqdm(factors):
        target_factor = load_factor(factor_name, start_date, end_date)
        target_factor = target_factor.groupby(level='code').ffill()
        if method == 'Residual':
            marginal_factor = get_residual(target_factor, bench_factors)
        else:
            marginal_factor = orthogonal_factor(target_factor, bench_factors, method=method)
        raw_analyzer = analyzer_cls(factor=target_factor[factor_name], prices=prices, periods=periods,
                                    groupby=groupby, weights=weights, **kwargs)
        marginal_analyzer = analyzer_cls(factor=marginal_factor[factor_name], prices=prices, periods=periods,
                                         groupby=groupby, weights=weights, **kwargs)

        # TODO: 并行
        raw_analyzer.create_report(factor_name)
        marginal_analyzer.create_report(f'{factor_name}_orth')

        # 对比报告：收益率、IC
        raw_returns = raw_analyzer.plot_returns_table(make_pretty=False)
        marginal_returns = marginal_analyzer.plot_returns_table(make_pretty=False)
        returns = pd.merge(raw_returns, marginal_returns, left_index=True, right_index=True,
                           suffixes=('_raw', '_orth'))
        raw_ic = raw_analyzer.plot_information_table(make_pretty=False)
        marginal_ic = marginal_analyzer.plot_information_table(make_pretty=False)
        ics = pd.merge(raw_ic, marginal_ic, left_index=True, right_index=True,
                           suffixes=('_raw', '_orth'))
        compare_df = pd.concat([returns, ics], axis=0)
        compare_df.to_csv(f'compares_{factor_name}.csv', index=True)
    return None


def analyze_account(account):
    """ account analysis based on empyrical """
    df = pd.DataFrame(account).set_index('date')[['asset']]
    nav = df['asset'] / df['asset'].iloc[0]
    returns = df['asset'].pct_change()
    ann_ret = ep.annual_return(returns)
    ann_vol = ep.annual_volatility(returns)
    mdd = ep.max_drawdown(returns)
    return {'nav': nav, 'ann_ret': ann_ret, 'ann_vol': ann_vol, 'mdd': mdd}

def backtests_report(cfg):
    if 'start_date' not in cfg:
        raise KeyError()

    start_date = cfg['start_date']
    end_date = cfg.get('end_date', date.today())

    loader = cfg.get('loader', 'mysql')
    loader_cls = loaders.get(loader, MysqlLoader)

    kwargs = cfg.copy()
    dataloader = loader_cls(**kwargs)

    bench_factors = cfg.get('bench_factors', config.style_factors)

    groupby = cfg.get('groupby', False)
    if groupby:
        groupby = dataloader.get_stock_industry(start_dt=start_date, end_dt=end_date)
        groupby.set_index(['date', 'code'], inplace=True)
    else:
        groupby = None

    weights = cfg.get('weights', False)
    if weights:
        weights = dataloader.get_industry_weight(start_dt=start_date, end_dt=end_date)
        weights.set_index(['date', 'group'], inplace=True)
    else:
        weights = 1.0

    bench_factors = dataloader.get_factor_data(bench_factors, start_dt=start_date, end_dt=end_date)
    bench_factors = bench_factors.groupby(level='code').ffill()
    prices = dataloader.get_stock_price(start_dt=start_date, end_dt=end_date)
    prices['date'] = pd.to_datetime(prices['date'].astype(str))

    bench_bt = Backtest(factors=bench_factors, prices=prices, group=groupby, group_weight=weights,
                        factor_weight=config.style_weights, initial_asset=10000000)
    bench_bt.run()
    perfs = {}
    perfs['benchmark'] = analyze_account(bench_bt.accounts)

    factors = list_factors()
    # TODO: 进程并行
    for factor_name in tqdm(factors):
        target_factor = load_factor(factor_name, start_date, end_date)
        target_factor = target_factor.groupby(level='code').ffill()

        factor_bt = Backtest(factors=bench_factors.merge(target_factor, left_index=True, right_index=True),
                             prices=prices, group=groupby, group_weight=weights, factor_weight=None,
                             initial_asset=10000000)
        factor_bt.run()
        perfs[factor_name] = analyze_account(factor_bt.accounts)
    return perfs


def backtests_report_by_industry(cfg):
    if 'start_date' not in cfg:
        raise KeyError()

    start_date = cfg['start_date']
    end_date = cfg.get('end_date', date.today())

    loader = cfg.get('loader', 'mysql')
    loader_cls = loaders.get(loader, MysqlLoader)

    kwargs = cfg.copy()
    dataloader = loader_cls(**kwargs)

    bench_factors = cfg.get('bench_factors', config.style_factors)

    groupby = dataloader.get_stock_industry(start_dt=start_date, end_dt=end_date)
    groupby = groupby.set_index(['date', 'code']).iloc[:, 0]

    bench_factors = dataloader.get_factor_data(bench_factors, start_dt=start_date, end_dt=end_date)
    bench_factors = bench_factors.groupby(level='code').ffill()
    prices = dataloader.get_stock_price(start_dt=start_date, end_dt=end_date)
    prices['date'] = pd.to_datetime(prices['date'].astype(str))

    perfs = {}
    factors = list_factors()

    for factor_name in tqdm(factors):
        target_factor = load_factor(factor_name, start_date, end_date)
        target_factor = target_factor.groupby(level='code').ffill()
        for sgn in [1, -1]:
            factor_weight = {factor_name: sgn}
            inds = groupby.unique()
            for ind in inds:
                tmp_grps = groupby[groupby==ind]
                tmp_stks = tmp_grps.index.get_level_values('code').unique()
                tmp_factor = target_factor[target_factor.index.get_level_values('code').isin(tmp_stks)]
                tmp_prices = prices[prices['code'].isin(tmp_stks)]
                factor_bt = Backtest(factors=tmp_factor, prices=tmp_prices, group=tmp_grps, group_weight=None,
                                     factor_weight=factor_weight, initial_asset=10000000)
                factor_bt.run()
                perfs[factor_name] = analyze_account(factor_bt.accounts)
    return perfs


if __name__ == '__main__':
    cfg = {'factor_name': 'beta', 'start_date': '20220101', 'end_date': '20240331',
           'loader': 'csv', 'data_path': Path(r'E:\DataRefactored'),
           'host': 'localhost', 'port': '3306',
           'username': 'wingtrade', 'password': 'wingtrade123', 'db': 'factor',
           'analyzer': 'classic', 'quantiles': 5}
    compares = alpha_report(cfg)