# -*- coding: utf-8 -*-
from collections.abc import Iterable
from functools import cached_property
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from .prepare import get_clean_factor_and_forward_returns
from .analyzer import FactorAnalyzer
from .utils import convert_to_forward_returns_columns, ensure_tuple


class ClassicAnalyzer(FactorAnalyzer):
    """ 经典因子分析，按quantile或bins分组
    """
    def __init__(
            self,
            factor: Union[pd.DataFrame, pd.Series],
            prices: Union[pd.DataFrame, Callable],
            groupby: Union[pd.DataFrame, Dict, Callable] = None,
            weights: Union[pd.DataFrame, Dict, Callable] = 1.0,
            quantiles: int = None,
            bins: int = None,
            periods: Tuple = (1, 5, 10),
            binning_by_group: bool = False,
            max_loss: float = 0.25,
            zero_aware: bool = False,
    ) -> None:
        self._quantiles = quantiles
        self._bins = bins
        super().__init__(factor, prices, groupby, weights, periods, binning_by_group, max_loss, zero_aware)
        self.__gen_clean_factor_and_forward_returns()

    def __gen_clean_factor_and_forward_returns(self):
        """格式化因子数据和定价数据"""

        factor_data: Union[pd.Series, pd.DataFrame] = self.factor
        if isinstance(factor_data, pd.DataFrame):
            # 仅当factor_data index-trade_date,columns-code,values-factors时适用
            factor_data: pd.Series = factor_data.stack(dropna=False)

        stocks: List[str] = (
            factor_data.index.get_level_values(1).drop_duplicates().tolist()
        )
        start_date: pd.Timestamp = factor_data.index.get_level_values(0).min()
        end_date: pd.Timestamp = factor_data.index.get_level_values(0).max()

        if hasattr(self.prices, "__call__"):
            prices: pd.DataFrame = self.prices(
                securities=stocks,
                start_date=start_date,
                end_date=end_date,
                period=max(self._periods),
            )
            prices: pd.DataFrame = prices.loc[~prices.index.duplicated()]
        else:
            prices: pd.DataFrame = self.prices
        self._prices: pd.DataFrame = prices

        if hasattr(self.groupby, "__call__"):
            groupby = self.groupby(
                securities=stocks, start_date=start_date, end_date=end_date
            )
        else:
            groupby = self.groupby
        self._groupby = groupby

        if hasattr(self.weights, "__call__"):
            weights = self.weights(stocks, start_date=start_date, end_date=end_date)
        else:
            weights = self.weights
        self._weights = weights

        self._clean_factor_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            factor_data,
            prices,
            groupby=groupby,
            weights=weights,
            binning_by_group=self._binning_by_group,
            quantiles=self._quantiles,
            bins=self._bins,
            periods=self._periods,
            max_loss=self._max_loss,
            zero_aware=self._zero_aware,
        )

    @property
    def _factor_quantile(self):
        data: pd.DataFrame = self.clean_factor_data
        if not data.empty:
            return data["factor_quantile"].max()
        else:
            _quantiles: int = self._quantiles
            _bins: int = self._bins
            _zero_aware: bool = self._zero_aware
            get_len = lambda x: len(x) - 1 if isinstance(x, Iterable) else int(x)
            if _quantiles is not None and _bins is None and not _zero_aware:
                return get_len(_quantiles)
            elif _quantiles is not None and _bins is None and _zero_aware:
                return int(_quantiles) // 2 * 2
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return get_len(_bins)
            elif _bins is not None and _quantiles is None and _zero_aware:
                return int(_bins) // 2 * 2

