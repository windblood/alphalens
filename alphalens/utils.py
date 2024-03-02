#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import re
import warnings

from IPython.display import display
from pandas.tseries.offsets import CustomBusinessDay, Day, BusinessDay
from scipy.stats import mode


class NonMatchingTimezoneError(Exception):
    pass


class MaxLossExceededError(Exception):
    pass


def rethrow(exception, additional_message):
    """
    Re-raise the last exception that was active in the current scope
    without losing the stacktrace but adding an additional message.
    This is hacky because it has to be compatible with both python 2/3
    """
    e = exception
    m = additional_message
    if not e.args:
        e.args = (m,)
    else:
        e.args = (e.args[0] + m,) + e.args[1:]
    raise e


def non_unique_bin_edges_error(func):
    """
    Give user a more informative error in case it is not possible
    to properly calculate quantiles on the input dataframe (factor)
    """
    message = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical
    values and they span more than one quantile. The quantiles are choosen
    to have the same number of records each, but the same value cannot span
    multiple quantiles. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get unequal
        number of records per quantile
    3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
        buckets to be evenly spaced according to the values themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - for factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value
    Please see utils.get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.

    """

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                rethrow(e, message)
            raise
    return dec


@non_unique_bin_edges_error
def cut_factor(
    factor_data,
    quantiles=5,
    bins=None,
    topK=None,
    topPercentage=None,
    by_group=False,
    no_raise=False,
    zero_aware=False,
):
    """
    计算每期因子分位数

    参数
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    quantiles : int or sequence[float]
        在因子分组中按照因子值大小平均分组的组数。
         或分位数序列, 允许不均匀分组
        例如 [0, .10, .5, .90, 1.] 或 [.05, .5, .95]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    bins : int or sequence[float]
        在因子分组中使用的等宽 (按照因子值) 区间的数量
        或边界值序列, 允许不均匀的区间宽度
        例如 [-4, -2, -0.5, 0, 10]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    topK: int, 按因子排序后选入的股票数量
    topPercentage: float, 按因子排序后选入的股票百分比——(0, 100)
    by_group : bool
        如果是 True, 按照 group 分别计算分位数
    no_raise: bool, optional
        如果为 True，则不抛出任何异常，并且将抛出异常的值设置为 np.NaN
    zero_aware : bool, optional
        如果为True，则分别为正负因子值计算分位数。
        适用于您的信号聚集并且零是正值和负值的分界线的情况.

    返回值
    -------
    factor_quantile : pd.Series
        index 为日期 (level 0) 和资产(level 1) 的因子分位数
    """
    none_num = sum(int(x is not None) for x in [quantiles, bins, topK, topPercentage])
    if not (none_num == 1):
        print("None_sum:", none_num)
        print("quantiles:", quantiles)
        print("bins:", bins)
        print("topK:", topK)
        print("topPercentage:", topPercentage)
        raise ValueError("[quantiles, bins, topPercentage, topK]中必须有且仅有一个不为None")

    if zero_aware and not (isinstance(quantiles, int) or isinstance(bins, int)):
        msg = "只有 quantiles 或 bins 为 int 类型时， 'zero_aware' 才能为 True"
        raise ValueError(msg)

    def cut_calc(x, _quantiles, _bins, _topK, _topPercentage, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = (
                    pd.qcut(x[x >= 0], _quantiles // 2, labels=False)
                    + _quantiles // 2
                    + 1
                )
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2, labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2, labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2, labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
            elif _topK is not None and _topPercentage is None:
                iorder = np.argsort(x.values)
                selected = np.ones_like(x.values)
                n = min(_topK, len(x))
                selected[iorder[-n:]] = 0
                res = pd.Series(selected, index=x.index) + 1
                return res
            elif _topK is None and _topPercentage is not None:
                iorder = np.argsort(x.values)
                selected = np.ones_like(x.values)
                n = int(_topPercentage/100 * len(x))
                selected[iorder[-n:]] = 0
                res = pd.Series(selected, index=x.index) + 1
                return res
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values("date")]
    if by_group:
        if "group" not in factor_data.columns:
            raise ValueError("只有输入了 groupby 参数时 binning_by_group 才能为 True")
        grouper.append("group")

    factor_quantile = factor_data.groupby(grouper, group_keys=False)["factor"].apply(
        cut_calc, quantiles, bins, topK, topPercentage, zero_aware, no_raise
    )
    factor_quantile.name = "factor_quantile"

    return factor_quantile.dropna()


@non_unique_bin_edges_error
def quantize_factor(factor_data,
                    quantiles=5,
                    bins=None,
                    by_group=False,
                    no_raise=False,
                    zero_aware=False):
    """
    Computes period wise factor quantiles.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group : bool, optional
        If True, compute quantile buckets separately for each group.
    no_raise: bool, optional
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.

    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and asset.
    """
    if not ((quantiles is not None and bins is None) or
            (quantiles is None and bins is not None)):
        raise ValueError('Either quantiles or bins should be provided')

    if zero_aware and not (isinstance(quantiles, int)
                           or isinstance(bins, int)):
        msg = ("zero_aware should only be True when quantiles or bins is an"
               " integer")
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = pd.qcut(x[x >= 0], _quantiles // 2,
                                        labels=False) + _quantiles // 2 + 1
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2,
                                        labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                  labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2,
                                  labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append('group')

    factor_quantile = factor_data.groupby(grouper, group_keys=False)['factor'] \
        .apply(quantile_calc, quantiles, bins, zero_aware, no_raise)

    factor_quantile.name = 'factor_quantile'

    return factor_quantile.dropna()


def infer_trading_calendar(factor_idx, prices_idx):
    """
    Infer the trading calendar from factor and price information.

    Parameters
    ----------
    factor_idx : pd.DatetimeIndex
        The factor datetimes for which we are computing the forward returns
    prices_idx : pd.DatetimeIndex
        The prices datetimes associated withthe factor data

    Returns
    -------
    calendar : pd.DateOffset
    """
    full_idx = factor_idx.union(prices_idx)

    traded_weekdays = []
    holidays = []

    days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day, day_str in enumerate(days_of_the_week):

        weekday_mask = (full_idx.dayofweek == day)

        # drop days of the week that are not traded at all
        if not weekday_mask.any():
            continue
        traded_weekdays.append(day_str)

        # look for holidays
        used_weekdays = full_idx[weekday_mask].normalize()
        all_weekdays = pd.date_range(full_idx.min(), full_idx.max(),
                                     freq=CustomBusinessDay(weekmask=day_str)
                                     ).normalize()
        _holidays = all_weekdays.difference(used_weekdays)
        _holidays = [timestamp.date() for timestamp in _holidays]
        holidays.extend(_holidays)

    traded_weekdays = ' '.join(traded_weekdays)
    return CustomBusinessDay(weekmask=traded_weekdays, holidays=holidays)


def compute_forward_returns(factor,
                            prices,
                            periods=(1, 5, 10),
                            filter_zscore=None,
                            cumulative_returns=True):
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property (forward_returns.index.levels[0].freq)
        will be set to a trading calendar (pandas DateOffset) inferred
        from the input data (see infer_trading_calendar for more details).
    """

    factor_dateindex = factor.index.levels[0]
    if factor_dateindex.tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and "
                                       "tz_convert.")

    freq = infer_trading_calendar(factor_dateindex, prices.index)

    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError("Factor and prices indices don't match: make sure "
                         "they have the same convention in terms of datetimes "
                         "and symbol-names")

    # chop prices down to only the assets we care about (= unique assets in
    # `factor`).  we could modify `prices` in place, but that might confuse
    # the caller.
    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    column_list = []

    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = \
            returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = np.nan

        #
        # Find the period length, which will be the column name. We'll test
        # several entries in order to find out the most likely period length
        # (in case the user passed inconsinstent data)
        #
        days_diffs = []
        for i in range(30):
            if i >= len(forward_returns.index):
                break
            p_idx = prices.index.get_loc(forward_returns.index[i])
            if p_idx is None or p_idx < 0 or (
                    p_idx + period) >= len(prices.index):
                continue
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = diff_custom_calendar_timedeltas(start, end, freq)
            days_diffs.append(period_len.components.days)

        delta_days = period_len.components.days - mode(days_diffs, keepdims=True).mode[0]
        period_len -= pd.Timedelta(days=delta_days)
        label = timedelta_to_string(period_len)

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns],
            names=['date', 'asset']
        ),
        inplace=True
    )
    df = df.reindex(factor.index)

    # now set the columns correctly
    df = df[column_list]

    try:
        df.index.levels[0].freq = freq
    except ValueError as e:
        print(e)
    df.index.set_names(['date', 'asset'], inplace=True)

    return df


def compute_backward_returns(factor,
                             prices,
                             periods=(1, 5, 10),
                             filter_zscore=None,
                             cumulative_returns=True):
    """
    Finds the N period backward returns (as percent change) for each asset
    provided.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property (forward_returns.index.levels[0].freq)
        will be set to a trading calendar (pandas DateOffset) inferred
        from the input data (see infer_trading_calendar for more details).
    """
    factor_dateindex = factor.index.levels[0]
    if factor_dateindex.tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and "
                                       "tz_convert.")

    freq = infer_trading_calendar(factor_dateindex, prices.index)
    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError("Factor and prices indices don't match: make sure "
                         "they have the same convention in terms of datetimes "
                         "and symbol-names")

    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    column_list = []

    for period in sorted(periods):
        returns = prices.pct_change(period) if cumulative_returns else prices.pct_change()
        returns = returns.reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(returns - returns.mean()) > (filter_zscore * returns.std())
            returns[mask] = np.nan

        days_diffs = []
        for i in range(30):
            if i >= len(returns.index):
                break
            p_idx = prices.index.get_loc(returns.index[i])
            if p_idx is None or p_idx < 0 or (
                    p_idx + period) >= len(prices.index):
                continue
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = diff_custom_calendar_timedeltas(start, end, freq)
            days_diffs.append(period_len.components.days)

        delta_days = period_len.components.days - mode(days_diffs, keepdims=True).mode[0]
        period_len -= pd.Timedelta(days=delta_days)
        label = timedelta_to_string(period_len)

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns],
            names=['date', 'asset']
        ),
        inplace=True
    )
    df = df.reindex(factor.index)
    df = df[column_list]
    df.index.levels[0].freq = freq
    df.index.set_names(['date', 'asset'], inplace=True)

    return df


def backshift_returns(returns, N):
    """Shift a multi-indexed returns backwards by N observations in
    the first level.

    This can be used to convert backward-looking returns into a
    forward-returns series.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise ValueError('returns should be pandas Series or DataFrame')
    ix = returns.index
    dates, sids = ix.levels
    date_codes, sid_codes = map(np.array, ix.codes)

    # Output date labels will contain the all but the last N dates.
    new_dates = dates[:-N]

    # Output data will remove the first M rows, where M is the index of the
    # last record with one of the first N dates.
    cutoff = date_codes.searchsorted(N)
    new_date_codes = date_codes[cutoff:] - N
    new_sid_codes = sid_codes[cutoff:]
    new_values = returns.values[cutoff:]

    assert new_date_codes[0] == 0

    new_index = pd.MultiIndex(
        levels=[new_dates, sids],
        codes=[new_date_codes, new_sid_codes],
        sortorder=1,
        names=ix.names,
    )
    if isinstance(returns, pd.Series):
        shifted_returns = pd.Series(data=new_values, index=new_index, name=returns.name)
    else:
        shifted_returns = pd.DataFrame(data=new_values, index=new_index, columns=returns.columns)

    return shifted_returns


def demean_forward_returns(factor_data, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows allows the
    factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe was 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(grouper, as_index=False)[
        cols.append(pd.Index(["weights"]))].apply(
        lambda x: x[cols].subtract(
            np.average(x[cols], axis=0, weights=x["weights"].fillna(0.0).values), axis=1
        )
    )

    return factor_data


def print_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def get_weight_type(weight_code, asset_code, group_code):
    # TODO
    return 'asset'


def get_clean_factor_own(
    factor,
    forward_returns,
    groupby = None,
    weights = None,
    binning_by_group = False,
    quantiles = 5,
    bins = None,
    topK = None,
    topPercentage=None,
    max_loss = 0.35,
    zero_aware = False,
) -> pd.DataFrame:
    """
    将因子值, 因子远期收益, 因子分组数据, 因子权重数据
    格式化为以时间和资产的 MultiIndex 作为索引的 DataFrame.

    参数
    ----------
    factor : pd.Series - MultiIndex
        一个 Series, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 为因子的值
    forward_returns : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 为因子的远期收益, columns 为因子远期收益的期数.
    groupby : pd.Series - MultiIndex or dict
        index 为日期和资产的 Series，为每个资产每天的分组，或资产-分组映射的字典.
        如果传递了dict，则假定分组映射在整个时间段内保持不变.
    weights : pd.Series - MultiIndex or dict
        index 为日期和资产的 Series，为每个资产每天的权重，或资产-权重映射的字典.
        如果传递了dict，则假定权重映射在整个时间段内保持不变.
        TODO: 为实现行业权重配置，添加支持group weight，根据index(level 1)取值，判断是资产权重还是group权重
        最终输出不变，保证每date每factor_quantile权重加总为1
    binning_by_group : bool
        如果为 True, 则对每个组分别计算分位数.
        适用于因子值范围在各个组上变化很大的情况.
        如果要分析分组(行业)中性的组合, 您最好设置为 True
    quantiles : int or sequence[float]
        在因子分组中按照因子值大小平均分组的组数。
         或分位数序列, 允许不均匀分组
        例如 [0, .10, .5, .90, 1.] 或 [.05, .5, .95]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    bins : int or sequence[float]
        在因子分组中使用的等宽 (按照因子值) 区间的数量
        或边界值序列, 允许不均匀的区间宽度
        例如 [-4, -2, -0.5, 0, 10]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    max_loss : float, optional
        允许的丢弃因子数据的最大百分比 (0.00 到 1.00),
        计算比较输入因子索引中的项目数和输出 DataFrame 索引中的项目数.
        因子数据本身存在缺陷 (例如 NaN),
        没有提供足够的价格数据来计算所有因子值的远期收益，
        或者因为分组失败, 因此可以部分地丢弃因子数据
        设置 max_loss = 0 以停止异常捕获.
    zero_aware : bool, optional
        如果为True，则分别为正负因子值计算分位数。
        适用于您的信号聚集并且零是正值和负值的分界线的情况.

    返回值
    -------
    merged_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
        - 各期因子远期收益的列名满足 'period_1', 'period_5' 的格式
    """

    initial_amount: int = len(factor.index)

    factor_copy: pd.DataFrame = factor.copy()
    factor_copy.index = factor_copy.index.rename(["date", "asset"])
    asset_code = factor_copy.index.get_level_values("asset")

    merged_data: pd.DataFrame = forward_returns.copy()
    merged_data["factor"] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(asset_code) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError("Assets {} not in group mapping".format(list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor_copy.index, data=ss[asset_code].values,)
        elif isinstance(groupby, pd.DataFrame):
            groupby = groupby.stack()
        merged_data["group"] = groupby
        group_code = set(groupby.values.tolist())

    if weights is not None:
        if isinstance(weights, dict):
            weight_code = set(weights.keys())
            diff = set(asset_code) - weight_code
            if len(diff) > 0:
                raise KeyError("Assets {} not in weights mapping".format(list(diff)))

            ww = pd.Series(weights)
            weights = pd.Series(
                index=factor_copy.index,
                data=ww[asset_code].values,
            )
        elif isinstance(weights, pd.DataFrame):
            weight_code = set(weights.index.get_level_values(-1))
            weights = weights.stack()
        if groupby is not None:
            weights_type = get_weight_type(weight_code, asset_code, group_code)
        else:
            weights_type = 'asset'
        if weights_type == 'asset':
            # 如果给定的是资产的权重，逻辑不变
            merged_data["weights"] = weights
        elif weights_type == 'group':
            # 如果给定的是group的权重
            group_weights = weights.reset_index()
            group_weights.columns = ['date', 'group', 'weights']
            merged_data['date'] = merged_data.index.get_level_values(level='date')
            merged_data = merged_data.merge(group_weights, on=['date', 'group'], how='left')

    merged_data = merged_data.dropna()

    # quantile_data = quantize_factor(
    #     merged_data, quantiles, bins, binning_by_group, True, zero_aware
    # )
    quantile_data = cut_factor(merged_data, quantiles, bins, topK, topPercentage, binning_by_group, True, zero_aware)

    merged_data["factor_quantile"] = quantile_data
    merged_data = merged_data.dropna()
    merged_data["factor_quantile"] = merged_data["factor_quantile"].astype(int)

    if "weights" in merged_data.columns:
        if weights_type == 'asset':
            merged_data["weights"] = (
                merged_data.set_index("factor_quantile", append=True)
                .groupby(level=["date", "factor_quantile"], group_keys=False)["weights"]
                .apply(lambda s: s.divide(s.sum()))
                .reset_index("factor_quantile", drop=True)
            )
        elif weights_type == 'group':
            # TODO: 如果有group weight，每天每个factor_quantile每个group加总weight应该是当天当group的weight
            # 待检验每天每个factor_quantile加总为1; binning_by_group分别为True/False下
            merged_data["weights"] = (
                merged_data.set_index("factor_quantile", append=True)
                .groupby(level=["date", "factor_quantile", "group"], group_keys=False)["weights"]
                .apply(lambda s: s / len(s))
                .reset_index("factor_quantile", drop=True)
            )

    binning_amount = float(len(merged_data.index))

    tot_loss = (initial_amount - binning_amount) / initial_amount

    no_raise = True if max_loss == 0 else False
    if tot_loss > max_loss and not no_raise:
        message = "max_loss (%.1f%%) 超过 %.1f%%" % (tot_loss * 100, max_loss * 100)
        raise MaxLossExceededError(message)

    return merged_data


def get_clean_factor_and_forward_returns_own(
    factor: pd.Series,
    prices: pd.DataFrame,
    groupby = None,
    weights = None,
    binning_by_group: bool = False,
    quantiles: int = 5,
    bins: int = None,
    topK: int = None,
    topPercentage: float=None,
    periods = (1, 5, 10),
    max_loss: float = 0.35,
    zero_aware: bool = False,
):
    """
    将因子数据, 价格数据, 分组映射和权重映射格式化为
    由包含时间和资产的 MultiIndex 作为索引的 DataFrame

    参数
    ----------
    factor : pd.Series - MultiIndex
     一个 Series, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 为因子的值
    prices : pd.DataFrame
        用于计算因子远期收益的价格数据
        columns 为资产, index 为 日期.
        价格数据必须覆盖因子分析时间段以及额外远期收益计算中的最大预期期数.
    groupby : pd.Series - MultiIndex or dict
        index 为日期和资产的 Series，为每个资产每天的分组，或资产-分组映射的字典.
        如果传递了dict，则假定分组映射在整个时间段内保持不变.
    weights : pd.Series - MultiIndex or dict
        index 为日期和资产的 Series，为每个资产每天的权重，或资产-权重映射的字典.
        如果传递了dict，则假定权重映射在整个时间段内保持不变.
    binning_by_group : bool
        如果为 True, 则对每个组分别计算分位数.
        适用于因子值范围在各个组上变化很大的情况.
        如果要分析分组(行业)中性的组合, 您最好设置为 True
    quantiles : int or sequence[float]
        在因子分组中按照因子值大小平均分组的组数。
         或分位数序列, 允许不均匀分组
        例如 [0, .10, .5, .90, 1.] 或 [.05, .5, .95]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    bins : int or sequence[float]
        在因子分组中使用的等宽 (按照因子值) 区间的数量
        或边界值序列, 允许不均匀的区间宽度
        例如 [-4, -2, -0.5, 0, 10]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    periods : sequence[int]
        远期收益的期数
    max_loss : float, optional
        允许的丢弃因子数据的最大百分比 (0.00 到 1.00),
        计算比较输入因子索引中的项目数和输出 DataFrame 索引中的项目数.
        因子数据本身存在缺陷 (例如 NaN),
        没有提供足够的价格数据来计算所有因子值的远期收益，
        或者因为分组失败, 因此可以部分地丢弃因子数据
        设置 max_loss = 0 以停止异常捕获.
    zero_aware : bool, optional
        如果为True，则分别为正负因子值计算分位数。
        适用于您的信号聚集并且零是正值和负值的分界线的情况.

    返回值
    -------
    merged_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
        - 各期因子远期收益的列名满足 'period_1', 'period_5' 的格式
    """

    forward_returns: pd.DataFrame = compute_forward_returns(factor, prices, periods)

    factor_data = get_clean_factor(
        factor,
        forward_returns,
        groupby=groupby,
        weights=weights,
        quantiles=quantiles,
        bins=bins,
        topK=topK,
        topPercentage=topPercentage,
        binning_by_group=binning_by_group,
        max_loss=max_loss,
        zero_aware=zero_aware,
    )

    return factor_data


def get_clean_factor(factor,
                     forward_returns,
                     groupby=None,
                     binning_by_group=False,
                     quantiles=5,
                     bins=None,
                     groupby_labels=None,
                     max_loss=0.35,
                     zero_aware=False):
    """
    Formats the factor data, forward return data, and group mappings into a
    DataFrame that contains aligned MultiIndex indices of timestamp and asset.
    The returned data will be formatted to be suitable for Alphalens functions.

    It is safe to skip a call to this function and still make use of Alphalens
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names must follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property must be set to a trading calendar
        (pandas DateOffset), see infer_trading_calendar for more details.
        This information is currently used only in cumulative returns
        computation
        ::
            ---------------------------------------
                       |       | 1D  | 5D  | 10D
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

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
        'quantiles' is None.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - forward returns column names follow the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)

        - 'date' index freq property (merged_data.index.levels[0].freq) is the
          same as that of the input forward returns data. This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------
    """

    initial_amount = float(len(factor.index))

    factor_copy = factor.copy()
    factor_copy.index = factor_copy.index.rename(['date', 'asset'])
    factor_copy = factor_copy[np.isfinite(factor_copy)]

    merged_data = forward_returns.copy()
    merged_data['factor'] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor_copy.index.get_level_values(
                'asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Assets {} not in group mapping".format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor_copy.index,
                                data=ss[factor_copy.index.get_level_values(
                                    'asset')].values)

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError(
                    "groups {} not in passed group names".format(
                        list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=groupby.index,
                                data=sn[groupby.values].values)

        # merged_data['group'] = groupby.astype('category')
        cat_dtype = pd.CategoricalDtype(categories=sorted(groupby.unique()), ordered=True)
        merged_data['group'] = groupby.astype(cat_dtype)

    merged_data = merged_data.dropna()

    fwdret_amount = float(len(merged_data.index))

    no_raise = False if max_loss == 0 else True
    quantile_data = quantize_factor(
        merged_data,
        quantiles,
        bins,
        binning_by_group,
        no_raise,
        zero_aware
    )

    merged_data['factor_quantile'] = quantile_data

    merged_data = merged_data.dropna()

    binning_amount = float(len(merged_data.index))

    tot_loss = (initial_amount - binning_amount) / initial_amount
    fwdret_loss = (initial_amount - fwdret_amount) / initial_amount
    bin_loss = tot_loss - fwdret_loss

    print("Dropped %.1f%% entries from factor data: %.1f%% in forward "
          "returns computation and %.1f%% in binning phase "
          "(set max_loss=0 to see potentially suppressed Exceptions)." %
          (tot_loss * 100, fwdret_loss * 100, bin_loss * 100))

    if tot_loss > max_loss:
        message = ("max_loss (%.1f%%) exceeded %.1f%%, consider increasing it."
                   % (max_loss * 100, tot_loss * 100))
        raise MaxLossExceededError(message)
    else:
        print("max_loss is %.1f%%, not exceeded: OK!" % (max_loss * 100))

    return merged_data


def get_ts_clean_factor(factor, forward_returns, groupby=None, binning_by_group=False, quantiles=5,
                        bins=None, groupby_labels=None, max_loss=0.35, zero_aware=False):
    """
    针对整个时序做分组并合并收益率，参数具体类型和含义参见get_clean_factor
    """

    def _quantile_factor_ts(factor_data, quantiles=5, bins=None, by_group=False,
                            no_raise=False, zero_aware=False):
        """Computes factor quantiles for whole time series """
        if not ((quantiles is not None and bins is None) or
                (quantiles is None and bins is not None)):
            raise ValueError("Either quantiles or bins should be provided.")

        if zero_aware and not (isinstance(quantiles, int)
                               or isinstance(bins, int)):
            msg = "zero_aware should only be True when quantils or bins is int"
            raise ValueError(msg)

        def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
            try:
                if _quantiles is not None and _bins is None and not _zero_aware:
                    return pd.qcut(x, _quantiles, labels=False) + 1
                elif _quantiles is not None and _bins is None and _zero_aware:
                    pos_quantiles = pd.qcut(x[x >= 0], _quantiles // 2,
                                            labels=False) + _quantiles // 2 +1
                    neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2,
                                            labels=False) + 1
                    return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
                elif _bins is not None and _quantiles is None and not _zero_aware:
                    return pd.cut(x, _bins, labels=False) + 1
                elif _bins is not None and _quantiles is None and _zero_aware:
                    pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                      labels=False) + _bins // 2 + 1
                    neg_bins = pd.cut(x[x < 0], _bins // 2,
                                      labels=False) + 1
                    return pd.concat([pos_bins, neg_bins]).sort_index()
            except Exception as e:
                if _no_raise:
                    return pd.Series(index=x.index)
                raise e

        if by_group:
            factor_quantile = factor_data.groupby('group', group_keys=False)['factor'] \
                               .apply(quantile_calc, quantiles, bins, zero_aware, no_raise)
        else:
            factor_quantile = quantile_calc(factor_data['factor'], quantiles, bins, zero_aware, no_raise)

        factor_quantile.name = 'factor_quantile'
        return factor_quantile.dropna()

    initial_amount = float(len(factor.index))

    factor_copy = factor.copy()
    factor_copy.index = factor_copy.index.rename(['date', 'asset'])
    factor_copy = factor_copy[np.isfinite(factor_copy)]

    merged_data = forward_returns.copy()
    merged_data['factor'] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor_copy.index.get_level_values(
                'asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Asset {} not in group mapping".format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor_copy.index,
                                data=ss[factor_copy.index.get_level_values(
                                    'asset')].values)

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError(
                    "group {} not in passed group names".format(
                        list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=groupby.index,
                                data=sn[groupby.values].values)

        # merged_data['group'] = groupby.astype('category')
        cat_dtype = pd.CategoricalDtype(categories=sorted(groupby.unique()), ordered=True)
        merged_data['group'] = groupby.astype(cat_dtype)

    merged_data = merged_data.dropna()
    fwdret_amount = float(len(merged_data.index))
    no_raise = False if max_loss == 0 else True
    quantile_data = _quantile_factor_ts(merged_data, quantiles, bins, binning_by_group,
                                        no_raise, zero_aware)

    merged_data['factor_quantile'] = quantile_data
    merged_data = merged_data.dropna()
    binning_amount = float(len(merged_data.index))

    tot_loss = (initial_amount - binning_amount) / initial_amount
    fwdret_loss = (initial_amount - fwdret_amount) / initial_amount
    bin_loss = tot_loss - fwdret_loss

    print("Dropped %.1f%% entries from factor data: %.1f%% in forward "
          "returns computation and %.1f%% in binning phase "
          "(set max_loss=0 to see potentially suppressed Exceptions). " %
          (tot_loss * 100, fwdret_loss * 100, bin_loss * 100))

    if tot_loss > max_loss:
        message = ("max_loss (%.1f%%) exceeded %.1f%%, considef increasing it."
                   % (max_loss * 100, tot_loss * 100))
        raise MaxLossExceededError(message)
    else:
        print("max_loss is %.1f%%, not exceeded: OK!" % (max_loss * 100))

    return merged_data


def get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=None,
                                         binning_by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         periods=(1, 5, 10),
                                         filter_zscore=20,
                                         groupby_labels=None,
                                         max_loss=0.35,
                                         zero_aware=False,
                                         cumulative_returns=True):
    """
    Formats the factor data, pricing data, and group mappings into a DataFrame
    that contains aligned MultiIndex indices of timestamp and asset. The
    returned data will be formatted to be suitable for Alphalens functions.

    It is safe to skip a call to this function and still make use of Alphalens
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns.
        Pricing data must span the factor analysis time period plus an
        additional buffer window that is greater than the maximum number
        of expected periods in the forward returns calculations.
        It is important to pass the correct pricing data in depending on
        what time of period your signal was generated so to avoid lookahead
        bias, or  delayed calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry should reflect the buy price
        for the assets and usually it is the next available price after the
        factor is computed but it can also be a later price if the factor is
        meant to be traded later (e.g. if the factor is computed at market
        open but traded 1 hour after market open the price information should
        be 1 hour after market open).
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        ::
            ----------------------------------------------------
                        | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
            ----------------------------------------------------
               Date     |      |      |       |       |        |
            ----------------------------------------------------
            2014-01-01  |605.12| 24.58|  11.72| 54.43 |  37.14 |
            ----------------------------------------------------
            2014-01-02  |604.35| 22.23|  12.21| 52.78 |  33.63 |
            ----------------------------------------------------
            2014-01-03  |607.94| 21.68|  14.36| 53.94 |  29.37 |
            ----------------------------------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)
        - 'date' index freq property (merged_data.index.levels[0].freq) will be
          set to a trading calendar (pandas DateOffset) inferred from the input
          data (see infer_trading_calendar for more details). This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------

    See Also
    --------
    utils.get_clean_factor
        For use when forward returns are already available.
    """
    forward_returns = compute_forward_returns(
        factor,
        prices,
        periods,
        filter_zscore,
        cumulative_returns,
    )

    factor_data = get_clean_factor(factor, forward_returns, groupby=groupby,
                                   groupby_labels=groupby_labels,
                                   quantiles=quantiles, bins=bins,
                                   binning_by_group=binning_by_group,
                                   max_loss=max_loss, zero_aware=zero_aware)

    return factor_data


def get_ts_clean_factor_and_forward_returns(factor,
                                            prices,
                                            groupby=None,
                                            binning_by_group=False,
                                            quantiles=5,
                                            bins=None,
                                            periods=(1, 5, 10),
                                            filter_zscore=20,
                                            groupby_labels=None,
                                            max_loss=0.35,
                                            zero_aware=False,
                                            cumulative_returns=True):
    """
    Formats the factor data, pricing data, and group mappings into a DataFrame
    that contains aligned MultiIndex indices of timestamp and asset. The
    returned data will be formatted to be suitable for Alphalens functions.

    It is safe to skip a call to this function and still make use of Alphalens
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns.
        Pricing data must span the factor analysis time period plus an
        additional buffer window that is greater than the maximum number
        of expected periods in the forward returns calculations.
        It is important to pass the correct pricing data in depending on
        what time of period your signal was generated so to avoid lookahead
        bias, or  delayed calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry should reflect the buy price
        for the assets and usually it is the next available price after the
        factor is computed but it can also be a later price if the factor is
        meant to be traded later (e.g. if the factor is computed at market
        open but traded 1 hour after market open the price information should
        be 1 hour after market open).
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        ::
            ----------------------------------------------------
                        | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
            ----------------------------------------------------
               Date     |      |      |       |       |        |
            ----------------------------------------------------
            2014-01-01  |605.12| 24.58|  11.72| 54.43 |  37.14 |
            ----------------------------------------------------
            2014-01-02  |604.35| 22.23|  12.21| 52.78 |  33.63 |
            ----------------------------------------------------
            2014-01-03  |607.94| 21.68|  14.36| 53.94 |  29.37 |
            ----------------------------------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)
        - 'date' index freq property (merged_data.index.levels[0].freq) will be
          set to a trading calendar (pandas DateOffset) inferred from the input
          data (see infer_trading_calendar for more details). This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------

    See Also
    --------
    utils.get_clean_factor
        For use when forward returns are already available.
    """
    forward_returns = compute_forward_returns(
        factor,
        prices,
        periods,
        filter_zscore,
        cumulative_returns,
    )

    factor_data = get_ts_clean_factor(factor, forward_returns, groupby=groupby,
                                      groupby_labels=groupby_labels,
                                      quantiles=quantiles, bins=bins,
                                      binning_by_group=binning_by_group,
                                      max_loss=max_loss, zero_aware=zero_aware)

    return factor_data


def get_clean_factor_and_backward_returns(factor,
                                         prices,
                                         groupby=None,
                                         binning_by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         periods=(1, 5, 10),
                                         filter_zscore=20,
                                         groupby_labels=None,
                                         max_loss=0.35,
                                         zero_aware=False,
                                         cumulative_returns=True):
    """
    Formats the factor data, pricing data, and group mappings into a DataFrame
    that contains aligned MultiIndex indices of timestamp and asset. The
    returned data will be formatted to be suitable for Alphalens functions.

    It is safe to skip a call to this function and still make use of Alphalens
    functionalities as long as the factor data conforms to the format returned
    from get_clean_factor_and_forward_returns and documented here

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
            -----------------------------------
                date    |    asset   |
            -----------------------------------
                        |   AAPL     |   0.5
                        -----------------------
                        |   BA       |  -1.1
                        -----------------------
            2014-01-01  |   CMG      |   1.7
                        -----------------------
                        |   DAL      |  -0.1
                        -----------------------
                        |   LULU     |   2.7
                        -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns.
        Pricing data must span the factor analysis time period plus an
        additional buffer window that is greater than the maximum number
        of expected periods in the forward returns calculations.
        It is important to pass the correct pricing data in depending on
        what time of period your signal was generated so to avoid lookahead
        bias, or  delayed calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry should reflect the buy price
        for the assets and usually it is the next available price after the
        factor is computed but it can also be a later price if the factor is
        meant to be traded later (e.g. if the factor is computed at market
        open but traded 1 hour after market open the price information should
        be 1 hour after market open).
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        ::
            ----------------------------------------------------
                        | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
            ----------------------------------------------------
               Date     |      |      |       |       |        |
            ----------------------------------------------------
            2014-01-01  |605.12| 24.58|  11.72| 54.43 |  37.14 |
            ----------------------------------------------------
            2014-01-02  |604.35| 22.23|  12.21| 52.78 |  33.63 |
            ----------------------------------------------------
            2014-01-03  |607.94| 21.68|  14.36| 53.94 |  29.37 |
            ----------------------------------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)
        - 'date' index freq property (merged_data.index.levels[0].freq) will be
          set to a trading calendar (pandas DateOffset) inferred from the input
          data (see infer_trading_calendar for more details). This is currently
          used only in cumulative returns computation
        ::
           -------------------------------------------------------------------
                      |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset |     |     |      |      |     |
           -------------------------------------------------------------------
                      | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                      --------------------------------------------------------

    See Also
    --------
    utils.get_clean_factor
        For use when forward returns are already available.
    """
    backward_returns = compute_backward_returns(
        factor,
        prices,
        periods,
        filter_zscore,
        cumulative_returns,
    )

    factor_data = get_clean_factor(factor, backward_returns, groupby=groupby,
                                   groupby_labels=groupby_labels,
                                   quantiles=quantiles, bins=bins,
                                   binning_by_group=binning_by_group,
                                   max_loss=max_loss, zero_aware=zero_aware)

    return factor_data


def rate_of_return(period_ret, base_period='1D'):
    """
    Convert returns to 'one_period_len' rate of returns: that is the value the
    returns would have every 'one_period_len' if they had grown at a steady
    rate

    Parameters
    ----------
    period_ret: pd.DataFrame
        DataFrame containing returns values with column headings representing
        the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc)

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with 'one_period_len' rate of
        returns values.
    """
    period_len = period_ret.name
    conversion_factor = (pd.Timedelta(base_period) /
                         pd.Timedelta(period_len))
    return period_ret.add(1).pow(conversion_factor).sub(1)


def std_conversion(period_std, base_period='1D'):
    """
    one_period_len standard deviation (or standard error) approximation

    Parameters
    ----------
    period_std: pd.DataFrame
        DataFrame containing standard deviation or standard error values
        with column headings representing the return period.
    base_period: string
        The base period length used in the conversion
        It must follow pandas.Timedelta constructor format (e.g. '1 days',
        '1D', '30m', '3h', '1D1h', etc)

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with one-period
        standard deviation/error values.
    """
    period_len = period_std.name
    conversion_factor = (pd.Timedelta(period_len) /
                         pd.Timedelta(base_period))
    return period_std / np.sqrt(conversion_factor)


def get_forward_returns_columns(columns, require_exact_day_multiple=False):
    """
    Utility that detects and returns the columns that are forward returns
    """

    # If exact day multiples are required in the forward return periods,
    # drop all other columns (e.g. drop 3D12h).
    if require_exact_day_multiple:
        pattern = re.compile(r"^(\d+([D]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

        if sum(valid_columns) < len(valid_columns):
            warnings.warn(
                "Skipping return periods that aren't exact multiples"
                + " of days."
            )
    else:
        pattern = re.compile(r"^(\d+([Dhms]|ms|us|ns]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

    return columns[valid_columns]


def get_forward_returns_columns_own(columns):
    syntax = re.compile("^period_\\d+$")
    return columns[columns.astype('str').str.contains(syntax, regex=True)]


def convert_to_forward_returns_columns(period):
    try:
        return 'period_{:d}'.format(period)
    except ValueError:
        return period


def timedelta_to_string(timedelta):
    """
    Utility that converts a pandas.Timedelta to a string representation
    compatible with pandas.Timedelta constructor format

    Parameters
    ----------
    timedelta: pd.Timedelta

    Returns
    -------
    string
        string representation of 'timedelta'
    """
    c = timedelta.components
    format = ''
    if c.days != 0:
        format += '%dD' % c.days
    if c.hours > 0:
        format += '%dh' % c.hours
    if c.minutes > 0:
        format += '%dm' % c.minutes
    if c.seconds > 0:
        format += '%ds' % c.seconds
    if c.milliseconds > 0:
        format += '%dms' % c.milliseconds
    if c.microseconds > 0:
        format += '%dus' % c.microseconds
    if c.nanoseconds > 0:
        format += '%dns' % c.nanoseconds
    return format


def timedelta_strings_to_integers(sequence):
    """
    Converts pandas string representations of timedeltas into integers of days.

    Parameters
    ----------
    sequence : iterable
        List or array of timedelta string representations, e.g. ['1D', '5D'].

    Returns
    -------
    sequence : list
        Integer days corresponding to the input sequence, e.g. [1, 5].
    """
    return list(map(lambda x: pd.Timedelta(x).days, sequence))


def add_custom_calendar_timedelta(input, timedelta, freq):
    """
    Add timedelta to 'input' taking into consideration custom frequency, which
    is used to deal with custom calendars, such as a trading calendar

    Parameters
    ----------
    input : pd.DatetimeIndex or pd.Timestamp
    timedelta : pd.Timedelta
    freq : pd.DataOffset (CustomBusinessDay, Day or BusinessDay)

    Returns
    -------
    pd.DatetimeIndex or pd.Timestamp
        input + timedelta
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BDay or CustomBusinessDay")
    days = timedelta.components.days
    offset = timedelta - pd.Timedelta(days=days)
    return input + freq * days + offset


def diff_custom_calendar_timedeltas(start, end, freq):
    """
    Compute the difference between two pd.Timedelta taking into consideration
    custom frequency, which is used to deal with custom calendars, such as a
    trading calendar

    Parameters
    ----------
    start : pd.Timestamp
    end : pd.Timestamp
    freq : CustomBusinessDay (see infer_trading_calendar)
    freq : pd.DataOffset (CustomBusinessDay, Day or BDay)

    Returns
    -------
    pd.Timedelta
        end - start
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BusinessDay or CustomBusinessDay")

    weekmask = getattr(freq, 'weekmask', None)
    holidays = getattr(freq, 'holidays', None)

    if weekmask is None and holidays is None:
        if isinstance(freq, Day):
            weekmask = 'Mon Tue Wed Thu Fri Sat Sun'
            holidays = []
        elif isinstance(freq, BusinessDay):
            weekmask = 'Mon Tue Wed Thu Fri'
            holidays = []

    if weekmask is not None and holidays is not None:
        # we prefer this method as it is faster
        actual_days = np.busday_count(np.array(start).astype('datetime64[D]'),
                                      np.array(end).astype('datetime64[D]'),
                                      weekmask, holidays)
    else:
        # default, it is slow
        actual_days = pd.date_range(start, end, freq=freq).shape[0] - 1
        if not freq.onOffset(start):
            actual_days -= 1

    timediff = end - start
    delta_days = timediff.components.days - actual_days
    return timediff - pd.Timedelta(days=delta_days)


PD_VERSION = pd.__version__


def rolling_apply(
    x,
    window,
    func,
    min_periods=None,
    freq=None,
    center=False,
    args=None,
    kwargs=None
):
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    if PD_VERSION >= '0.23.0':
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).apply(
            func, False, args=args, kwargs=kwargs
        )
    elif PD_VERSION >= '0.18.0':
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).apply(
            func, args=args, kwargs=kwargs
        )
    else:
        return pd.rolling_apply(
            x,
            window,
            func,
            min_periods=min_periods,
            freq=freq,
            center=center,
            args=args,
            kwargs=kwargs
        )


def rolling_mean(x, window, min_periods=None, center=False):
    if PD_VERSION >= '0.18.0':
        return x.rolling(window, min_periods=min_periods, center=center).mean()
    else:
        return pd.rolling_mean(
            x, window, min_periods=min_periods, center=center
        )


def rolling_std(x, window, min_periods=None, center=False, ddof=1):
    if PD_VERSION >= '0.18.0':
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).std(ddof=ddof)
    else:
        return pd.rolling_std(
            x, window, min_periods=min_periods, center=center, ddof=ddof
        )