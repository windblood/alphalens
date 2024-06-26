# -*- coding: utf-8 -*-
from collections.abc import Iterable
from functools import cached_property
from typing import Callable, Dict, List, Tuple, Union
from abc import abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.figure_factory import create_table
from functools import lru_cache
from scipy import stats
from scipy.stats import pearsonr, spearmanr

import alphalens.performance as perf
import alphalens.plotting as pl
from alphalens.plotly_plotting import _use_chinese

from alphalens.utils import (get_clean_factor_and_forward_returns, convert_to_forward_returns_columns,
                     rate_of_return, std_conversion)


class FactorAnalyzer(object):
    def __init__(
        self,
        factor: Union[pd.DataFrame, pd.Series],
        prices: Union[pd.DataFrame, Callable],
        groupby: Union[pd.DataFrame, Dict, Callable] = None,
        weights: Union[pd.DataFrame, Dict, Callable] = 1.0,
        periods: Tuple = (1, 5, 10),
        binning_by_group: bool = False,
        max_loss: float = 0.25,
        zero_aware: bool = False,
    ) -> None:
        self.factor = factor
        self.prices = prices
        self.groupby = groupby
        self.weights = weights
        self._periods = periods

        self._binning_by_group = binning_by_group
        self._max_loss = max_loss
        self._zero_aware = zero_aware
        _use_chinese(True)
        print("get_clear_data.......")
        self._prices = None
        self._clean_factor_data = None
        # self.__gen_clean_factor_and_forward_returns()

    @abstractmethod
    def __gen_clean_factor_and_forward_returns(self):
        """格式化因子数据和定价数据"""
        raise NotImplementedError("子类需要实现__gen_clean_factor_and_forward_returns，"
                                  "生成有效的_prices和_clean_factor_data")

    @property
    def clean_factor_data(self):
        return self._clean_factor_data

    @property
    def _factor_quantile(self):
        data: pd.DataFrame = self.clean_factor_data
        if not data.empty:
            return data["factor_quantile"].max()
        else:
            _quantiles: int = self._quantiles
            _bins: int = self._bins
            _zero_aware: bool = self._zero_aware
            get_len: int = lambda x: len(x) - 1 if isinstance(x, Iterable) else int(x)
            if _quantiles is not None and _bins is None and not _zero_aware:
                return get_len(_quantiles)
            elif _quantiles is not None and _bins is None and _zero_aware:
                return int(_quantiles) // 2 * 2
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return get_len(_bins)
            elif _bins is not None and _quantiles is None and _zero_aware:
                return int(_bins) // 2 * 2

    @lru_cache(16)
    def calc_mean_return_by_quantile(
            self, by_date=False, by_group=False, demeaned=False, group_adjust=False
    ):
        """计算按分位数分组因子收益和标准差

        因子收益为收益按照 weight 列中权重的加权平均值

        参数:
        by_date:
        - True: 按天计算收益
        - False: 不按天计算收益
        by_group:
        - True: 按行业计算收益
        - False: 不按行业计算收益
        demeaned:
        - True: 使用超额收益计算各分位数收益，超额收益=收益-基准收益
                (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益计算各分位数收益，行业中性收益=收益-行业收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        return perf.mean_return_by_quantile(
            self._clean_factor_data,
            by_date=by_date,
            by_group=by_group,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

    @lru_cache(4)
    def calc_factor_returns(self, demeaned=True, group_adjust=False):
        """计算按因子值加权组合每日收益

        权重 = 每日因子值 / 每日因子值的绝对值的和
        正的权重代表买入, 负的权重代表卖出

        参数:
        demeaned:
        - True: 对权重去均值 (每日权重 = 每日权重 - 每日权重的均值), 使组合转换为 cash-neutral 多空组合
        - False: 不对权重去均值
        group_adjust:
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        return perf.factor_returns(
            self._clean_factor_data, demeaned=demeaned, group_adjust=group_adjust
        )

    def compute_mean_returns_spread(
            self,
            upper_quant=None,
            lower_quant=None,
            by_date=True,
            by_group=False,
            demeaned=False,
            group_adjust=False,
    ):
        """计算两个分位数相减的因子收益和标准差

        参数:
        upper_quant: 用 upper_quant 选择的分位数减去 lower_quant 选择的分位数
        lower_quant: 用 upper_quant 选择的分位数减去 lower_quant 选择的分位数
        by_date:
        - True: 按天计算两个分位数相减的因子收益和标准差
        - False: 不按天计算两个分位数相减的因子收益和标准差
        by_group:
        - True: 分行业计算两个分位数相减的因子收益和标准差
        - False: 不分行业计算两个分位数相减的因子收益和标准差
        demeaned:
        - True: 使用超额收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        """
        upper_quant = upper_quant if upper_quant is not None else self._factor_quantile
        lower_quant = lower_quant if lower_quant is not None else 1
        if (not 1 <= upper_quant <= self._factor_quantile) or (
                not 1 <= lower_quant <= self._factor_quantile
        ):
            raise ValueError(
                "upper_quant 和 low_quant 的取值范围为 1 - %s 的整数" % self._factor_quantile
            )
        mean, std = self.calc_mean_return_by_quantile(
            by_date=by_date,
            by_group=by_group,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )
        mean = mean.apply(rate_of_return, axis=0)
        std = std.apply(std_conversion, axis=0)
        return perf.compute_mean_returns_spread(
            mean_returns=mean,
            upper_quant=upper_quant,
            lower_quant=lower_quant,
            std_err=std,
        )

    @lru_cache(4)
    def calc_factor_alpha_beta(self, demeaned=True, group_adjust=False):
        """计算因子的 alpha 和 beta

        因子值加权组合每日收益 = beta * 市场组合每日收益 + alpha

        因子值加权组合每日收益计算方法见 calc_factor_returns 函数
        市场组合每日收益是每日所有股票收益按照weight列中权重加权的均值
        结果中的 alpha 是年化 alpha

        参数:
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值),
                使组合转换为cash-neutral多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        return perf.factor_alpha_beta(
            self._clean_factor_data, demeaned=demeaned, group_adjust=group_adjust
        )

    @lru_cache(8)
    def calc_factor_information_coefficient(
            self, group_adjust=False, by_group=False, method=None
    ):
        """计算每日因子信息比率 (IC值)

        参数:
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        if method is None:
            method = "rank"
        if method not in ("rank", "normal"):
            raise ValueError("`method` should be chosen from ('rank' | 'normal')")

        if method == "rank":
            method = spearmanr
        elif method == "normal":
            method = pearsonr
        return perf.factor_information_coefficient(
            self._clean_factor_data,
            group_adjust=group_adjust,
            by_group=by_group,
            method=method,
        )

    @lru_cache(16)
    def calc_mean_information_coefficient(
            self, group_adjust=False, by_group=False, by_time=None, method=None
    ):
        """计算因子信息比率均值 (IC值均值)

        参数:
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        by_time:
        - 'Y': 按年求均值
        - 'M': 按月求均值
        - None: 对所有日期求均值
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        if method is None:
            method = "rank"
        if method not in ("rank", "normal"):
            raise ValueError("`method` should be chosen from ('rank' | 'normal')")

        if method == "rank":
            method = spearmanr
        elif method == "normal":
            method = pearsonr
        return perf.mean_information_coefficient(
            self._clean_factor_data,
            group_adjust=group_adjust,
            by_group=by_group,
            by_time=by_time,
            method=method,
        )

    @lru_cache(16)
    def calc_average_cumulative_return_by_quantile(
            self, periods_before, periods_after, demeaned=False, group_adjust=False
    ):
        """按照当天的分位数算分位数未来和过去的收益均值和标准差

        参数:
        periods_before: 计算过去的天数
        periods_after: 计算未来的天数
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        return perf.average_cumulative_return_by_quantile(
            self._clean_factor_data,
            prices=self._prices,
            periods_before=periods_before,
            periods_after=periods_after,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

    @lru_cache(2)
    def calc_autocorrelation(self, rank=True):
        """根据调仓周期确定滞后期的每天计算因子自相关性

        当日因子值和滞后period天的因子值的自相关性

        参数:
        rank:
        - True: 秩相关系数
        - False: 普通相关系数
        """
        return pd.concat(
            [
                perf.factor_autocorrelation(self._clean_factor_data, period, rank=rank)
                for period in self._periods
            ],
            axis=1,
            keys=list(map(convert_to_forward_returns_columns, self._periods)),
        )

    @lru_cache(None)
    def calc_quantile_turnover_mean_n_days_lag(self, n=10):
        """各分位数滞后1天到n天的换手率均值

        参数:
        n: 滞后 1 天到 n 天的换手率
        """
        quantile_factor = self._clean_factor_data["factor_quantile"]

        quantile_turnover_rate = pd.concat(
            [
                pd.Series(
                    [
                        perf.quantile_turnover(quantile_factor, q, p).mean()
                        for q in range(1, int(quantile_factor.max()) + 1)
                    ],
                    index=range(1, int(quantile_factor.max()) + 1),
                    name=p,
                )
                for p in range(1, n + 1)
            ],
            axis=1,
            keys="lag_" + pd.Index(range(1, n + 1)).astype(str),
        ).T
        quantile_turnover_rate.columns.name = "factor_quantile"

        return quantile_turnover_rate

    @lru_cache(None)
    def calc_autocorrelation_n_days_lag(self, n=10, rank=False):
        """滞后1-n天因子值自相关性

        参数:
        n: 滞后1天到n天的因子值自相关性
        rank:
        - True: 秩相关系数
        - False: 普通相关系数
        """
        return pd.Series(
            [
                perf.factor_autocorrelation(self._clean_factor_data, p, rank=rank).mean()
                for p in range(1, n + 1)
            ],
            index="lag_" + pd.Index(range(1, n + 1)).astype(str),
        )

    @lru_cache(None)
    def _calc_ic_mean_n_day_lag(
            self, n, group_adjust=False, by_group=False, method=None
    ):
        if method is None:
            method = "rank"
        if method not in ("rank", "normal"):
            raise ValueError("`method` should be chosen from ('rank' | 'normal')")

        if method == "rank":
            method = spearmanr
        elif method == "normal":
            method = pearsonr

        factor_data = self._clean_factor_data.copy()
        factor_value = factor_data["factor"].unstack("asset")

        factor_data["factor"] = factor_value.shift(n).stack(dropna=True)
        if factor_data.dropna().empty:
            return pd.Series(
                np.nan, index=perf.get_forward_returns_columns(factor_data.columns)
            )
        ac = perf.factor_information_coefficient(
            factor_data.dropna(),
            group_adjust=group_adjust,
            by_group=by_group,
            method=method,
        )
        return ac.mean(level=("group" if by_group else None))

    def calc_ic_mean_n_days_lag(
            self, n=10, group_adjust=False, by_group=False, method=None
    ):
        """滞后 0 - n 天因子收益信息比率(IC)的均值

        滞后 n 天 IC 表示使用当日因子值和滞后 n 天的因子收益计算 IC

        参数:
        n: 滞后0-n天因子收益的信息比率(IC)的均值
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        ic_mean = [
            self.calc_factor_information_coefficient(
                group_adjust=group_adjust,
                by_group=by_group,
                method=method,
            ).mean(level=("group" if by_group else None))
        ]

        for lag in range(1, n + 1):
            ic_mean.append(
                self._calc_ic_mean_n_day_lag(
                    n=lag, group_adjust=group_adjust, by_group=by_group, method=method
                )
            )
        if not by_group:
            ic_mean = pd.concat(
                ic_mean, keys="lag_" + pd.Index(range(n + 1)).astype(str), axis=1
            )
            ic_mean = ic_mean.T
        else:
            ic_mean = pd.concat(
                ic_mean, keys="lag_" + pd.Index(range(n + 1)).astype(str), axis=0
            )
        return ic_mean

    @property
    def mean_return_by_quantile(self):
        """收益分析

        用来画分位数收益的柱状图

        返回 pandas.DataFrame, index 是 factor_quantile, 值是(1, 2, 3, 4, 5),
        column 是 period 的值 (1, 5, 10)
        """
        mean_ret_quantile, _ = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=False,
            demeaned=False,
            group_adjust=False,
        )
        mean_compret_quantile = mean_ret_quantile.apply(rate_of_return, axis=0)
        return mean_compret_quantile

    @property
    def mean_return_std_by_quantile(self):
        """收益分析

        用来画分位数收益的柱状图

        返回 pandas.DataFrame, index 是 factor_quantile, 值是(1, 2, 3, 4, 5),
        column 是 period 的值 (1, 5, 10)
        """
        _, mean_ret_std_quantile = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=False,
            demeaned=False,
            group_adjust=False,
        )
        mean_ret_std_quantile = mean_ret_std_quantile.apply(std_conversion, axis=0)
        return mean_ret_std_quantile

    @property
    def _mean_return_by_date(self):
        _mean_return_by_date, _ = self.calc_mean_return_by_quantile(
            by_date=True,
            by_group=False,
            demeaned=False,
            group_adjust=False,
        )
        return _mean_return_by_date

    @property
    def mean_return_by_date(self):
        mean_return_by_date = self._mean_return_by_date.apply(rate_of_return, axis=0)
        return mean_return_by_date

    @property
    def mean_return_std_by_date(self):
        _, std_quant_daily = self.calc_mean_return_by_quantile(
            by_date=True,
            demeaned=False,
            by_group=False,
            group_adjust=False,
        )
        mean_return_std_by_date = std_quant_daily.apply(std_conversion, axis=0)

        return mean_return_std_by_date

    @property
    def mean_return_by_group(self):
        """分行业的分位数收益

        返回值:
            MultiIndex 的 DataFrame
            index 分别是分位数、 行业名称,  column 是 period  (1, 5, 10)
        """
        mean_return_group, _ = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=True,
            demeaned=True,
            group_adjust=False,
        )
        mean_return_group = mean_return_group.apply(rate_of_return, axis=0)
        return mean_return_group

    @property
    def mean_return_std_by_group(self):
        _, mean_return_std_group = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=True,
            demeaned=True,
            group_adjust=False,
        )
        mean_return_std_group = mean_return_std_group.apply(std_conversion, axis=0)
        return mean_return_std_group

    @property
    def mean_return_spread_by_quantile(self):
        mean_return_spread_by_quantile, _ = self.compute_mean_returns_spread()
        return mean_return_spread_by_quantile

    @property
    def mean_return_spread_std_by_quantile(self):
        _, std_spread_quant = self.compute_mean_returns_spread()
        return std_spread_quant

    @lru_cache(5)
    def calc_cumulative_return_by_quantile(
            self, period=None, demeaned=False, group_adjust=False
    ):
        """计算指定调仓周期的各分位数每日累积收益

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        if period is None:
            period = self._periods[0]
        period_col = convert_to_forward_returns_columns(period)

        factor_returns = self.calc_mean_return_by_quantile(
            by_date=True, demeaned=demeaned, group_adjust=group_adjust
        )[0][period_col].unstack("factor_quantile")

        cum_ret = factor_returns.apply(perf.cumulative_returns, period=period)

        return cum_ret

    @lru_cache(20)
    def calc_cumulative_returns(self, period=None, demeaned=False, group_adjust=False) -> pd.Series:
        """计算指定调仓周期的按因子值加权组合每日累积收益

        当 period > 1 时，组合的累积收益计算方法为：
        组合每日收益 = （从第0天开始每period天一调仓的组合每日收益 +
                        从第1天开始每period天一调仓的组合每日收益 + ... +
                        从第period-1天开始每period天一调仓的组合每日收益) / period
        组合累积收益 = 组合每日收益的累积

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对权重去均值 (每日权重 = 每日权重 - 每日权重的均值), 使组合转换为 cash-neutral 多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        if period is None:
            period: int = self._periods[0]
        period_col: str = convert_to_forward_returns_columns(period)
        factor_returns: pd.Series = self.calc_factor_returns(
            demeaned=demeaned, group_adjust=group_adjust
        )[period_col]

        return perf.cumulative_returns(factor_returns, period=period)

    @lru_cache(5)
    def calc_top_down_returns(
            self,
            period: int = None,
            demeaned: bool = False,
            group_adjust: bool = False,
            reversed: bool = False,
    ) -> pd.Series:
        if period is None:
            period: int = self._periods[0]
        period_col: str = convert_to_forward_returns_columns(period)
        mean_returns, _ = self.calc_mean_return_by_quantile(
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

        upper_quant: pd.Series = mean_returns[period_col].xs(
            self._factor_quantile, level="factor_quantile"
        )
        lower_quant: pd.Series = mean_returns[period_col].xs(1, level="factor_quantile")
        if reversed:
            return lower_quant - lower_quant

        return upper_quant - lower_quant

    @lru_cache(20)
    def calc_top_down_cumulative_returns(
            self,
            period: int = None,
            demeaned: bool = False,
            group_adjust: bool = False,
            reversed: bool = False,
    ) -> pd.Series:
        """计算做多最大分位，做空最小分位组合每日累积收益

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        diff_ser: pd.Series = self.calc_top_down_returns(period=period, demeaned=demeaned,
                                                         group_adjust=group_adjust, reversed=reversed,)

        return perf.cumulative_returns(diff_ser, period=period)

    @property
    def ic(self):
        """IC 分析, 日度 ic

        返回 DataFrame, index 是时间,  column 是 period 的值 (1, 5, 10)
        """
        return self.calc_factor_information_coefficient()

    @property
    def ic_by_group(self):
        """行业 ic"""
        return self.calc_mean_information_coefficient(by_group=True)

    @property
    def ic_monthly(self):
        ic_monthly = self.calc_mean_information_coefficient(
            group_adjust=False, by_group=False, by_time="M"
        ).copy()
        ic_monthly.index = ic_monthly.index.map(lambda x: x.strftime("%Y-%m"))
        return ic_monthly

    @cached_property
    def quantile_turnover(self):
        """换手率分析

        返回值一个 dict, key 是 period, value 是一个 DataFrame(index 是日期, column 是分位数)
        """

        quantile_factor = self._clean_factor_data["factor_quantile"]

        quantile_turnover_rate = {
            convert_to_forward_returns_columns(p): pd.concat(
                [
                    perf.quantile_turnover(quantile_factor, q, p)
                    for q in range(1, int(quantile_factor.max()) + 1)
                ],
                axis=1,
            )
            for p in self._periods
        }

        return quantile_turnover_rate

    @property
    def cumulative_return_by_quantile(self):
        return {
            convert_to_forward_returns_columns(
                p
            ): self.calc_cumulative_return_by_quantile(period=p)
            for p in self._periods
        }

    @property
    def cumulative_returns(self):
        return pd.concat(
            [self.calc_cumulative_returns(period=period) for period in self._periods],
            axis=1,
            keys=list(map(convert_to_forward_returns_columns, self._periods)),
        )

    @property
    def top_down_cumulative_returns(self):
        return pd.concat(
            [
                self.calc_top_down_cumulative_returns(period=period)
                for period in self._periods
            ],
            axis=1,
            keys=list(map(convert_to_forward_returns_columns, self._periods)),
        )

    @property
    def top_down_returns(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.calc_top_down_returns(period=period)
                for period in self._periods
            ],
            axis=1,
            keys=list(map(convert_to_forward_returns_columns, self._periods)),
        )

    def plot_returns_table(
            self, demeaned=False, group_adjust=False, make_pretty: bool = True
    ):
        """打印因子收益表

        参数:
        demeaned:
        - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        mean_return_by_quantile = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )[0].apply(rate_of_return, axis=0)

        mean_returns_spread, _ = self.compute_mean_returns_spread(
            upper_quant=self._factor_quantile,
            lower_quant=1,
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

        returns_table: pd.DataFrame = pl.get_returns_table(
            self.calc_factor_alpha_beta(demeaned=demeaned),
            mean_return_by_quantile,
            mean_returns_spread,
        )

        if make_pretty:
            if returns_table.shape[1] > 1:
                returns_table = returns_table.pipe(pl.pretty_comparison)

            return returns_table.pipe(pl.set_table_title, title="收益分析")

        return returns_table

    def plot_turnover_table(self, make_pretty: bool = True):
        """打印换手率表"""
        turnover_table, auto_corr = pl.get_turnover_table(
            self.calc_autocorrelation(), self.quantile_turnover
        )

        if make_pretty:
            if turnover_table.shape[1] > 1:
                turnover_table, auto_corr = turnover_table.pipe(
                    pl.make_background_gradient
                ), auto_corr.pipe(pl.pretty_comparison)

            return turnover_table.pipe(
                pl.set_table_title, title="换手率分析-分组均值"
            ), auto_corr.pipe(pl.set_table_title, title="换手率分析-滞后期自相关系数")

        return turnover_table, auto_corr

    def plot_tstats_table(self, make_pretty: bool = True):
        """打印 t 统计量表"""
        ttest_frame: pd.DataFrame = perf.calculate_factor_ttest(self._clean_factor_data)
        ttest_table: pd.DataFrame = pl.get_tstats_table(ttest_frame)

        if make_pretty:
            if ttest_table.shape[1] > 1:
                ttest_table = ttest_table.pipe(pl.pretty_comparison)

            return ttest_table.pipe(pl.set_table_title, title="t 统计量分析")

        return ttest_table

    def plot_information_table(
            self, group_adjust=False, method=None, make_pretty: bool = True
    ):
        """打印信息比率 (IC)相关表

        参数:
        group_adjust:
        - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False：不使用行业中性收益
        method：
        - 'rank'：用秩相关系数计算IC值
        - 'normal':用相关系数计算IC值
        """
        ic: pd.DataFrame = self.calc_factor_information_coefficient(
            group_adjust=group_adjust, by_group=False, method=method
        )

        ic_table: pd.DataFrame = pl.get_information_table(ic)

        if make_pretty:
            if ic_table.shape[1] > 1:
                ic_table = ic_table.pipe(pl.pretty_comparison)
            return ic_table.pipe(pl.set_table_title, title="IC 分析")

        return ic_table

    def plot_quantile_statistics_table(self, make_pretty: bool = True):
        """打印各分位数统计表"""

        quantile_statistics_table: pd.DataFrame = pl.get_quantile_statistics_table(
            self._clean_factor_data
        )

        if make_pretty:
            return quantile_statistics_table.style.pipe(
                pl.set_table_title, title="分位数统计"
            )
        return quantile_statistics_table

    def plot_ic_ts(self, group_adjust: bool = False, method: str = None):
        """画信息比率(IC)时间序列图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal':用相关系数计算IC值
        """
        ic: pd.DataFrame = self.calc_factor_information_coefficient(
            group_adjust=group_adjust, by_group=False, method=method
        )
        return pl.plot_ic_ts(ic)

    def plot_cumulative_ic_ts(self, group_adjust: bool = False, method: str = None):
        ic: pd.DataFrame = self.calc_factor_information_coefficient(
            group_adjust=group_adjust, by_group=False, method=method
        )

        return pl.plot_cumulative_ic_ts(ic)

    def plot_ic_hist(self, group_adjust=False, method=None):
        """画信息比率分布直方图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        """
        ic: pd.DataFrame = self.calc_factor_information_coefficient(
            group_adjust=group_adjust, by_group=False, method=method
        )
        return pl.plot_ic_hist(ic)

    def plot_ic_qq(
            self, group_adjust=False, method=None, theoretical_dist=None
    ) -> List[go.Figure]:
        """画信息比率 qq 图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        theoretical_dist:
        - 'norm': 正态分布
        - 't': t 分布
        """
        theoretical_dist: str = "norm" if theoretical_dist is None else theoretical_dist
        theoretical_dist = getattr(stats, theoretical_dist)
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust,
            by_group=False,
            method=method,
        )
        return pl.plot_ic_qq(ic, theoretical_dist=theoretical_dist)

    def plot_quantile_returns_bar(
            self, by_group=False, demeaned=False, group_adjust=False
    ) -> go.Figure:
        """画各分位数平均收益图

        参数:
        by_group:
        - True: 各行业的各分位数平均收益图
        - False: 各分位数平均收益图
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        mean_return_by_quantile = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=by_group,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )[0].apply(rate_of_return, axis=0)

        return pl.plot_quantile_returns_bar(
            mean_return_by_quantile, by_group=by_group, ylim_percentiles=None
        )

    def plot_quantile_returns_violin(
            self, demeaned=False, group_adjust=False, ylim_percentiles=(1, 99)
    ) -> go.Figure:
        """画各分位数收益分布图

        参数:
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        plot_quantile_returns_violin: 有效收益分位数(单位为百分之). 画图时y轴的范围为有效收益的最大/最小值.
                                      例如 (1, 99) 代表收益的从小到大排列的 1% 分位到 99% 分位为有效收益.
        """
        mean_return_by_date = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False, demeaned=demeaned, group_adjust=group_adjust
        )[0].apply(rate_of_return, axis=0)

        return pl.plot_quantile_returns_violin(
            mean_return_by_date, ylim_percentiles=ylim_percentiles
        )

    def plot_mean_quantile_returns_spread_time_series(
            self, demeaned=False, group_adjust=False, bandwidth=1
    ):
        """画最高分位减最低分位收益图

        参数:
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        bandwidth: n, 加减 n 倍当日标准差
        """
        mean_returns_spread, mean_returns_spread_std = self.compute_mean_returns_spread(
            upper_quant=self._factor_quantile,
            lower_quant=1,
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

        return pl.plot_mean_quantile_returns_spread_time_series(
            mean_returns_spread, std_err=mean_returns_spread_std, bandwidth=bandwidth
        )

    def plot_factor_auto_correlation(self, periods=None, rank=True):
        """画因子自相关图

        参数:
        periods: 滞后周期
        rank:
        - True: 用秩相关系数
        - False: 用相关系数
        """
        periods: Tuple = tuple(self._periods if periods is None else periods)
        autocorrelation = self.calc_autocorrelation(rank=rank)

        figs = [
            pl.plot_factor_rank_auto_correlation(
                autocorrelation[convert_to_forward_returns_columns(p)], period=p
            )
            for p in periods
            if p in self._periods
        ]

        return figs

    def plot_top_bottom_quantile_turnover(self, periods=None) -> List[go.Figure]:
        """画最高最低分位换手率图

        参数:
        periods: 调仓周期
        """
        quantile_turnover = self.quantile_turnover
        periods = tuple(self._periods if periods is None else periods)

        figs = [
            pl.plot_top_bottom_quantile_turnover(
                quantile_turnover[convert_to_forward_returns_columns(p)], period=p
            )
            for p in periods
            if p in self._periods
        ]

        return figs

    def plot_monthly_ic_heatmap(self, group_adjust=False) -> List[go.Figure]:
        """画月度信息比率(IC)图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        ic_monthly = self.calc_mean_information_coefficient(
            group_adjust=group_adjust, by_group=False, by_time="M"
        )
        return pl.plot_monthly_ic_heatmap(ic_monthly)

    def plot_cumulative_returns(
            self, period=None, demeaned=False, group_adjust=False
    ):
        """画按因子值加权组合每日累积收益图

        参数:
        periods: 调仓周期
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值),
                使组合转换为cash-neutral多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        period = tuple(self._periods if period is None else period)
        if not isinstance(period, Iterable):
            period = (period,)

        period = tuple(period)
        factor_returns = self.calc_factor_returns(
            demeaned=demeaned, group_adjust=group_adjust
        )

        figs = [
            pl.plot_cumulative_returns(
                factor_returns[convert_to_forward_returns_columns(p)], period=p
            )
            for p in period
            if p in self._periods
        ]

        return figs

    def plot_top_down_cumulative_returns(
            self,
            period: int = None,
            demeaned: bool = False,
            group_adjust: bool = False,
            reversed: bool = False,
    ):
        """画做多最大分位数做空最小分位数组合每日累积收益图

        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益

        reversed:逆转因子值排序进行计算
        """
        period = tuple(self._periods if period is None else period)

        figs = [
            pl.plot_top_down_cumulative_returns(
                self.calc_top_down_cumulative_returns(
                    period=p,
                    demeaned=demeaned,
                    group_adjust=group_adjust,
                    reversed=reversed,
                ),
                period=p,
            )
            for p in period
            if p in self._periods
        ]

        return figs

    def plot_cumulative_returns_by_quantile(
            self, period: int = None, demeaned: bool = False, group_adjust: bool = False
    ) -> List[go.Figure]:
        """画各分位数每日累积收益图

        参数:
        period: 调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        period = tuple(self._periods if period is None else period)
        mean_return_by_date, _ = self.calc_mean_return_by_quantile(
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

        figs = [
            pl.plot_cumulative_returns_by_quantile(
                mean_return_by_date[convert_to_forward_returns_columns(p)], period=p
            )
            for p in period
            if p in self._periods
        ]

        return figs

    def plot_quantile_average_cumulative_return(
            self,
            periods_before=5,
            periods_after=10,
            by_quantile=False,
            std_bar=False,
            demeaned=False,
            group_adjust=False,
    ) -> go.Figure:
        """因子预测能力平均累计收益图

        参数:
        periods_before: 计算过去的天数
        periods_after: 计算未来的天数
        by_quantile: 是否各分位数分别显示因子预测能力平均累计收益图
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        average_cumulative_return_by_q = self.calc_average_cumulative_return_by_quantile(
            periods_before=periods_before,
                periods_after=periods_after,
                demeaned=demeaned,
                group_adjust=group_adjust,)
        average_cumulative_return_by_q: pd.DataFrame = (
            average_cumulative_return_by_q.sort_index(axis=1)
        )
        return pl.plot_quantile_average_cumulative_return(
            average_cumulative_return_by_q,
            by_quantile=by_quantile,
            std_bar=std_bar,
            periods_before=str(periods_before),
            periods_after=str(periods_after),
        )

    def plot_events_distribution(self, num_days=5):
        """画有效因子数量统计图

        参数:
        num_days: 统计间隔天数
        """
        if isinstance(self.factor, pd.DataFrame):
            dates = pd.to_datetime(self.factor.index.unique())
        elif isinstance(self.factor, pd.Series):
            self.factor.index.names = ["date", "asset"]
            dates = pd.to_datetime(self.factor.index.get_level_values("date").unique())
        pl.plot_events_distribution(
            events=self._clean_factor_data["factor"],
            num_days=num_days,
            full_dates=dates,
        )

    def create_report(self, name, demeaned=False, group_adjust=False) -> None:
        """ 输出因子报告

        参数:
        demeaned:
        - True: 对每日因子收益去均值求得因子收益表
        - False: 因子收益表
        group_adjust:
        - True: 按行业对因子收益去均值后求得因子收益表
        - False: 因子收益表
        """
        with open(f'{name}.html', 'a', encoding='utf8') as f:
            # 因子各分位统计表
            tbl = self.plot_quantile_statistics_table(make_pretty=False)
            tbl = create_table(tbl, index=True, index_title='因子分位数统计')
            f.write(pio.to_html(tbl, full_html=False))
            # tbl = self.plot_quantile_statistics_table(make_pretty=True)
            # f.write(tbl.to_html(header=False))
            f.write('<br>')
            f.write('<br>')
            f.write('<br>')

            # 因子收益表
            # tbl = self.plot_returns_table(
            #     demeaned=demeaned, group_adjust=group_adjust, make_pretty=False
            # )
            # tbl = create_table(tbl, index=True, index_title='因子收益表')
            # f.write(pio.to_html(tbl, full_html=False))
            tbl = self.plot_returns_table(
                demeaned=demeaned, group_adjust=group_adjust, make_pretty=True
            )
            f.write(tbl.to_html(header=False))
            f.write('<br>')
            f.write('<br>')
            f.write('<br>')

            # 因子收益率分布图
            fig = self.plot_quantile_returns_bar(
                by_group=False, demeaned=demeaned, group_adjust=group_adjust
            )
            f.write(pio.to_html(fig, full_html=False))

            # 因子值加权组合每日累积收益图
            fig = self.plot_cumulative_returns(
                period=None, demeaned=demeaned, group_adjust=group_adjust
            )
            for fig_ in fig:
                f.write(pio.to_html(fig_, full_html=False))

            # 分位数每日累积收益图
            fig = self.plot_cumulative_returns_by_quantile(
                period=None, demeaned=demeaned, group_adjust=group_adjust
            )
            for fig_ in fig:
                f.write(pio.to_html(fig_, full_html=False))

            # 做多最大分位数做空最小分位数组合每日累积收益图
            fig = self.plot_top_down_cumulative_returns(
                period=None, demeaned=demeaned, group_adjust=group_adjust
            )
            for fig_ in fig:
                f.write(pio.to_html(fig_, full_html=False))

            # 高分位减最低分位收益图
            fig = self.plot_mean_quantile_returns_spread_time_series(
                demeaned=demeaned, group_adjust=group_adjust
            )
            for fig_ in fig:
                f.write(pio.to_html(fig_, full_html=False))

            # 各分位数收益分布图
            fig = self.plot_quantile_returns_violin(
                demeaned=demeaned, group_adjust=group_adjust
            )
            f.write(pio.to_html(fig, full_html=False))

            # 信息比率(IC)相关表
            tbl = self.plot_information_table(group_adjust=group_adjust, make_pretty=True)
            # tbl = create_table(tbl, index=True, index_title='信息比率表')
            # f.write(pio.to_html(tbl, full_html=False))
            f.write(tbl.to_html(header=False))

            fig = self.plot_ic_ts(group_adjust=group_adjust, method=None)
            if isinstance(fig, list):
                for fig_ in fig:
                    f.write(pio.to_html(fig_, full_html=False))
            else:
                f.write(pio.to_html(fig, full_html=False))
            fig = self.plot_cumulative_ic_ts(group_adjust=group_adjust)
            f.write(pio.to_html(fig, full_html=False))

            # t 统计表
            tbl = self.plot_tstats_table(make_pretty=False)
            tbl = create_table(tbl, index=True, index_title='t统计表')
            f.write(pio.to_html(tbl, full_html=False))

            fig = self.plot_top_bottom_quantile_turnover(periods=self._periods)
            for fig_ in fig:
                f.write(pio.to_html(fig_, full_html=False))

            # 换手率表-分组均值; 换手率表-滞后期自相关系数
            turnover_table, auto_corr = self.plot_turnover_table(make_pretty=False)
            turnover_table = create_table(turnover_table, index=True, index_title='换手率表')
            f.write(pio.to_html(turnover_table, full_html=False))
            auto_corr = create_table(auto_corr, index=True, index_title='自相关表')
            f.write(pio.to_html(auto_corr, full_html=False))

            figs = self.plot_factor_auto_correlation(periods=self._periods)
            for fig in figs:
                f.write(pio.to_html(fig, full_html=False))


#     def create_summary_tear_sheet(self, demeaned=False, group_adjust=False) -> None:
#         """因子值特征分析
#
#         参数:
#         demeaned:
#         - True: 对每日因子收益去均值求得因子收益表
#         - False: 因子收益表
#         group_adjust:
#         - True: 按行业对因子收益去均值后求得因子收益表
#         - False: 因子收益表
#         """
#
#         with st.container():
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("**因子各分位统计表**")
#                 st.dataframe(self.plot_quantile_statistics_table())
#
#             with col2:
#                 st.markdown("**因子收益表**")
#                 st.dataframe(
#                     self.plot_returns_table(
#                         demeaned=demeaned, group_adjust=group_adjust
#                     )
#                 )
#
#         st.markdown("**因子收益率分布图**")
#
#         plotting_by_streamlit(
#             self.plot_quantile_returns_bar(
#                 by_group=False, demeaned=demeaned, group_adjust=group_adjust
#             ),
#         )
#
#         with st.container():
#             col3, col4 = st.columns(2)
#
#             with col3:
#                 st.markdown("**信息比率(IC)相关表**")
#                 st.dataframe(self.plot_information_table(group_adjust=group_adjust))
#
#             with col4:
#                 st.markdown("**t 统计量表**")
#                 st.dataframe(self.plot_tstats_table())
#
#             turnover_table, auto_corr = self.plot_turnover_table()
#             col5, col6 = st.columns(2)
#             with col5:
#                 st.markdown("**换手率表-分组均值**")
#                 st.dataframe(turnover_table)
#
#             with col6:
#                 st.markdown("**换手率表-滞后期自相关系数**")
#
#                 st.dataframe(auto_corr)
#
#     def create_returns_tear_sheet(
#             self, demeaned: bool = False, group_adjust: bool = False, by_group: bool = False
#     ) -> None:
#         """因子值特征分析
#
#         参数:
#         demeaned:
#         详见 calc_mean_return_by_quantile 中 demeaned 参数
#         - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
#         - False: 不使用超额收益
#         group_adjust:
#         详见 calc_mean_return_by_quantile 中 group_adjust 参数
#         - True: 使用行业中性化后的收益计算 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
#         - False: 不使用行业中性化后的收益
#         by_group:
#         - True: 画各行业的各分位数平均收益图
#         - False: 不画各行业的各分位数平均收益图
#         """
#
#         st.markdown("**因子收益表**")
#         st.dataframe(
#             self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
#         )
#
#         st.markdown("**分位数平均收益图**")
#
#         plotting_by_streamlit(
#             self.plot_quantile_returns_bar(
#                 by_group=False, demeaned=demeaned, group_adjust=group_adjust
#             )
#         )
#
#         st.markdown("**因子值加权组合每日累积收益图**")
#         plotting_in_grid(
#             self.plot_cumulative_returns(
#                 period=None, demeaned=demeaned, group_adjust=group_adjust
#             )
#         )
#
#         st.markdown("**分位数每日累积收益图**")
#         plotting_in_grid(
#             self.plot_cumulative_returns_by_quantile(
#                 period=None, demeaned=demeaned, group_adjust=group_adjust
#             )
#         )
#
#         st.markdown("**做多最大分位数做空最小分位数组合每日累积收益图**")
#         plotting_in_grid(
#             self.plot_top_down_cumulative_returns(
#                 period=None, demeaned=demeaned, group_adjust=group_adjust
#             )
#         )
#
#         st.markdown("**高分位减最低分位收益图**")
#         plotting_by_streamlit(
#             self.plot_mean_quantile_returns_spread_time_series(
#                 demeaned=demeaned, group_adjust=group_adjust
#             )
#         )
#
#         st.markdown("**各分位数收益分布图**")
#         plotting_by_streamlit(
#             self.plot_quantile_returns_violin(
#                 demeaned=demeaned, group_adjust=group_adjust
#             )
#         )
#
#     def create_information_tear_sheet(self, group_adjust=False, by_group=False):
#         """因子 IC 分析
#
#         参数:
#         group_adjust:
#         - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
#         - False: 不使用行业中性收益
#         by_group:
#         - True: 画按行业分组信息比率(IC)图
#         - False: 画月度信息比率(IC)图
#         """
#         plotting_by_streamlit(self.plot_ic_ts(group_adjust=group_adjust, method=None))
#
#         plotting_by_streamlit(self.plot_cumulative_ic_ts(group_adjust=group_adjust))
#
#         plotting_in_grid(self.plot_ic_qq(group_adjust=group_adjust))
#
#         # TODO:尚未实现
#         # if by_group:
#         #     self.plot_ic_by_group(group_adjust=group_adjust, method=None)
#         # else:
#         #     self.plot_monthly_ic_heatmap(group_adjust=group_adjust)
#         plotting_in_grid(self.plot_monthly_ic_heatmap(group_adjust=group_adjust))
#
#     def create_turnover_tear_sheet(self, turnover_periods=None):
#         """因子换手率分析
#
#         参数:
#         turnover_periods: 调仓周期
#         """
#         st.markdown("**换手率**")
#
#         turnover_table, auto_corr = self.plot_turnover_table()
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("**换手率表-分组均值**")
#             st.dataframe(turnover_table)
#
#         with col2:
#             st.markdown("**换手率表-滞后期自相关系数**")
#
#             st.dataframe(auto_corr)
#
#         st.markdown("**最高最低分位换手率**")
#
#         plotting_in_grid(
#             self.plot_top_bottom_quantile_turnover(periods=turnover_periods)
#         )
#
#         st.markdown("**因子自相关图**")
#         plotting_in_grid(self.plot_factor_auto_correlation(periods=turnover_periods))
#
#     def create_event_returns_tear_sheet(
#             self, avgretplot=(5, 15), demeaned=False, group_adjust=False, std_bar=True
#     ):
#         """因子预测能力分析
#
#         参数:
#         avgretplot: tuple 因子预测的天数
#         -(计算过去的天数, 计算未来的天数)
#         demeaned:
#         详见 calc_mean_return_by_quantile 中 demeaned 参数
#         - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
#         - False: 不使用超额收益
#         group_adjust:
#         详见 calc_mean_return_by_quantile 中 group_adjust 参数
#         - True: 使用行业中性化后的收益计算累积收益
#                 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
#         - False: 不使用行业中性化后的收益
#         std_bar:
#         - True: 显示标准差
#         - False: 不显示标准差
#         """
#         before, after = avgretplot
#         plotting_by_streamlit(
#             self.plot_quantile_average_cumulative_return(
#                 periods_before=before,
#                 periods_after=after,
#                 by_quantile=False,
#                 std_bar=False,
#                 demeaned=demeaned,
#                 group_adjust=group_adjust,
#             )
#         )
#
#         if std_bar:
#             plotting_by_streamlit(
#                 self.plot_quantile_average_cumulative_return(
#                     periods_before=before,
#                     periods_after=after,
#                     by_quantile=True,
#                     std_bar=True,
#                     demeaned=demeaned,
#                     group_adjust=group_adjust,
#                 )
#             )
#
#
# def create_full_tear_sheet(factor_analyze):
#         factor_analyze.create_summary_tear_sheet()
#         factor_analyze.create_returns_tear_sheet()
#         factor_analyze.create_information_tear_sheet()
#         factor_analyze.create_turnover_tear_sheet()
#         factor_analyze.create_event_returns_tear_sheet()
