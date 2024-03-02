# -*- coding: utf-8 -*-
from collections.abc import Iterable
from functools import cached_property
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd


from analyzer.plot_utils import _use_chinese, plotting_by_streamlit, plotting_in_grid
from analyzer.prepare import get_clean_factor_and_forward_returns
from analyzer.streamlit_analyze import FactorAnalyzer
from analyzer.utils import convert_to_forward_returns_columns, ensure_tuple


class TopKAnalyzer(FactorAnalyzer):
    """top k选股的效果(ZY)
    原本的quantiles改为topK为组1，其他为组2
    """
    def __init__(
            self,
            factor: Union[pd.DataFrame, pd.Series],
            prices: Union[pd.DataFrame, Callable],
            groupby: Union[pd.DataFrame, Dict, Callable] = None,
            weights: Union[pd.DataFrame, Dict, Callable] = 1.0,
            topK: int = 3,
            periods: Tuple = (1, 5, 10),
            binning_by_group: bool = False,
            max_loss: float = 0.25,
            zero_aware: bool = False,
    ) -> None:
        self._topK = topK
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
            quantiles=None,
            topK=self._topK,
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
            return 2


if __name__ == '__main__':
    from data_service import MysqlLoader
    from analyzer import create_full_tear_sheet
    loader = MysqlLoader(host='localhost', port='3306', username='wingtrade',
                         password='wingtrade123', db='factor')
    factor_list = loader.get_factor_name_list
    factor_data = loader.get_factor_data(factor_names=factor_list[:2], start_dt='20190101', end_dt='20231231')
    factor_data['trade_date'] = pd.to_datetime(factor_data['trade_date'].astype(str))
    factor_name = factor_list[0]
    factor_ser = (factor_data.set_index(["trade_date", "code"])
                  .query("factor_name==@factor_name")["value"]
                  .sort_index()
                  .dropna())
    codes = [x for x in factor_data["code"].unique().tolist() if x is not None]
    ind_df = loader.get_stock_industry(codes=codes, start_dt='20190101', end_dt='20240217')
    ind_dict = dict(zip(ind_df['code'], ind_df['ind_code']))
    default_dict = {x: '015001' for x in codes if x not in ind_dict}
    ind_dict.update(default_dict)
    full_codes = [x + '.SH' if x.startswith('6') else x + '.SZ' for x in codes]
    stock_price = loader.get_stock_price(codes=full_codes, start_dt='20190101', end_dt='20231231')
    stock_price['trade_date'] = pd.to_datetime(stock_price['trade_date'].astype(str))
    pricing: pd.DataFrame = pd.pivot_table(
        stock_price, index="trade_date", columns="code", values='close'
    )

    topK = 5
    factor_analyzer = TopKAnalyzer(factor_ser, prices=pricing, groupby=ind_dict, binning_by_group=True,
                                   topK=topK, periods=(1,5,10), max_loss=0.98)
    create_full_tear_sheet(factor_analyzer, factor_name)