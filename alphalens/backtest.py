#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
desc: 多因子组合模块

factors: pd.DataFrame with multiindex of (date, asset) , columns of factors
prices: pd.DataFrame / Series of stock prices, with multiindex of (date, asset);
        使用时选择用到的价格，转为宽dataframe(date x asset)
group: optional, pd.Series with multiindex of (date, asset), group as value
group_weight: optional, pd.Series with multiindex of (date, group), weight as value
factor_weight: optional, pd.Series with multiindex of (date, factor), weight as value

TODO:用config文件指定因子列表、时间、股票范围、权重、参数，自动加载数据，回测


@author: admin
@date: 2024/2/29
"""
from functools import reduce
from typing import List
import pickle

import numpy as np
import pandas as pd


def load_pickle(file):
    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res

def save_pickle(file, accounts):
    with open(file, 'wb') as f:
        pickle.dump(accounts, f)
    return None

def market_value(port_dict, price_dict):
    value = sum([vol*price_dict[stk] for stk, vol in port_dict.items()])
    return value


def gen_trades(pre_port, new_port):
    pre_stk = set(pre_port.keys())
    new_stk = set(new_port.keys())
    buy = new_stk - pre_stk
    buy = {stk: new_port[stk] for stk in buy}
    sell = pre_stk - new_stk
    sell = {stk: -pre_port[stk] for stk in sell}
    adjust = new_stk.intersection(pre_stk)
    adjust = {stk: new_port[stk] - pre_port[stk] for stk in adjust}
    res = {}
    res.update(sell)
    res.update(buy)
    res.update(adjust)
    return res


class Backtest:
    def __init__(self, factors, prices, group=None, group_weight=None, factor_weight=None, **kwargs):
        self.factors = factors
        col = kwargs.get('price_type', 'close')
        self.prices = prices.set_index(['date', 'code'])[col]  # 外面可以穿入long df，选择价格col后应该转为wide df，后向填充价格(不处理停牌)
        self.dates = self.prices.index.get_level_values('date').unique().sort_values()
        date_range = self.dates[[0, -1]].astype(str).to_list()
        self.initial_asset = kwargs.get('initial_asset', 10000000)

        self.group = self.fill_dates(group)  # 按self.dates填充group、group_weight、factor_weight
        self.group_weight = self.fill_dates(group_weight)
        self.factor_weight = self.fill_dates(factor_weight)
        self.name = '_'.join(self.factors.columns.to_list() + date_range)

    def fill_dates(self, sr):
        if sr is None:
            return None
        if isinstance(sr, dict) or (isinstance(sr, pd.Series) and sr.index.ndim == 1):
            sr_index = sr.index if isinstance(sr, pd.Series) else list(sr.keys())
            df = pd.DataFrame(index=self.dates, columns=sr_index)
            for idx in sr_index:
                df[idx] = sr[idx]
            sr = df
            sr = sr.melt(var_name='group', value_name='value', ignore_index=False).set_index('group', append=True)
            sr = sr['value']
        elif isinstance(sr, pd.Series) and sr.index.ndim > 1:
            value_name = sr.name
            col_name = sr.index.names[-1]
            sr = sr.reset_index().pivot(index='date', columns=col_name, values=value_name)
            sr = sr.reindex(self.dates).ffill().bfill()
            sr = sr.melt(var_name=col_name, value_name=value_name, ignore_index=False).set_index(col_name, append=True)
            sr = sr[value_name]
        else:
            return sr
        return sr

    def is_rebalance(self, holding_account, rebalance_period=9):
        if holding_account == 0:
            return True
        if isinstance(rebalance_period, int) and holding_account == rebalance_period:
            return True
        return False

    def save(self, path):
        # save params
        # save accounts
        if 'accounts' in dir(self):
            save_pickle(path / f'{self.name}.pickle', self.accounts)

    def run(self, load=False, **kwargs):
        """ 功能优化
        1. account设计，支持热加载
        2. 简单算法，向量化运行
        """
        dates = self.dates
        rebalance_period = kwargs.get('rebalance', 9)
        cost = kwargs.get('cost', 0)
        is_hedge = kwargs.get('hedge', False)

        # 初始化/加载 账户
        if not load:
            start = 0
            holding_count = 0
            accounts = []
            pre_account = {}
        else:
            load_path = kwargs.get('file', f'{self.name}.pickle')
            accounts = load_pickle(load_path)
            pre_account = accounts[-1]
            holding_count = pre_account['holding_count']
            last_date = pre_account['date']
            start = dates.index(last_date) + 1

        for dt in dates[start:]:
            pre_asset = pre_account.get('asset', self.initial_asset)
            pre_cash = pre_account.get('cash', self.initial_asset)
            pre_portfolio = pre_account.get('portfolio', {})

            tmp_prices = self.prices.loc[dt]
            prices_dict = tmp_prices.to_dict()
            old_mkt_value = market_value(pre_portfolio, prices_dict)
            is_rebalance = self.is_rebalance(holding_count, rebalance_period)
            if is_rebalance:
                # 调仓日，生成组合
                tmp_factors = self.factors.loc[dt]
                tmp_group = self.group.loc[dt] if self.group is not None else None
                tmp_group_weight = self.group_weight.loc[dt] if self.group_weight is not None else None
                tmp_factor_weight = self.factor_weight.loc[dt] if self.factor_weight is not None else None

                portfolio_weight = self.gen_portfolio(tmp_factors, tmp_prices, tmp_group,
                                               tmp_group_weight, tmp_factor_weight, pre_account)

                portfolio = {stk: round(pre_asset*(1-cost*2)*wgt/prices_dict[stk], -2)
                             for stk, wgt in portfolio_weight.items()}
                trades = gen_trades(pre_portfolio, portfolio)
                fee = sum([abs(vol) * prices_dict[stk]*cost for stk, vol in trades.items()])
                mkt_value = market_value(portfolio, prices_dict)
                cash = pre_cash + old_mkt_value - fee - mkt_value
            else:
                # 非调仓日，更新账户
                portfolio = pre_portfolio.copy()
                mkt_value = old_mkt_value
                cash = pre_cash

            asset = mkt_value + cash

            # 记录昨日信息
            if is_rebalance:
                holding_count = 1 if len(portfolio) > 0 else 0
            else:
                holding_count += 1
            pre_asset = asset
            pre_cash = cash
            pre_portfolio = portfolio.copy()
            pre_account = {'date': dt, 'asset': pre_asset, 'cash': pre_cash, 'portfolio': pre_portfolio,
                           'holding_count': holding_count}
            accounts.append(pre_account)

        self.accounts = accounts

    def gen_portfolio(self, factors, prices=None, group=None, group_weight=None, factor_weight=None,
                      account=None, **kwargs):
        """ 单期组合 """
        # factors, index: asset, column: factor, value: factor value
        if group is None:
            group = pd.Series(1, index=factors.index)  # index: asset, value: group
        groups = group.unique()

        if group_weight is None:
            group_weight = 1  # index: group, value: weight
        else:
            if len(groups) > 1:
                group_weight = group_weight.reindex(groups).fillna(0)

        if factor_weight is None:
            factor_weight = pd.Series(1, index=factors.columns)  # index: factor, value: weight

        selections = {}
        topk = kwargs.get('topK', 3)
        for grp in groups:
            tmp_group = group[group == grp]
            tmp_factor = factors.reindex(tmp_group.index)   # nan process
            tmp_score = tmp_factor.rank(pct=True)
            weighted_score = pd.Series(tmp_score.values @ factor_weight.values, index=tmp_score.index)
            tmp_selection = weighted_score.sort_values().index[:topk]
            tmp_group_weight = group_weight if isinstance(group_weight, (int, float)) else group_weight.loc[grp]
            tmp_weights = tmp_group_weight / topk
            selections.update({stk: tmp_weights for stk in tmp_selection})

        total_weights = sum(selections.values())
        selections = {stk: wgt/total_weights for stk, wgt in selections.items()}
        return selections


if __name__ == '__main__':
    from pathlib import Path
    from loader import CSVLoader
    start_date = '20200101'
    end_date = '20231231'
    factors = ['turn_10d', 'BP', 'beta', 'con_roe_roll']
    loader = CSVLoader(data_path=Path(r'E:\DataRefactored'))
    factors = loader.get_factor_data(factor_names=factors, codes=None, start_dt=start_date, end_dt=end_date)
    prices = loader.get_stock_price(start_dt=start_date, end_dt=end_date)
    # groups = loader.get_stock_industry(start_dt=start_date, end_dt=end_date)
    # groups = groups.set_index(['date', 'code'])['group']
    # group_weights = loader.get_industry_weight(start_dt=start_date, end_dt=end_date)
    # group_weights = group_weights.set_index(['date', 'group'])['weight']
    groups, group_weights = None,None
    factor_weights = None
    bt = Backtest(factors=factors, prices=prices, group=groups, group_weight=group_weights,
                  factor_weight=factor_weights, initial_asset=10000000)
    bt.run(load=False,)
    bt.save(Path('./'))