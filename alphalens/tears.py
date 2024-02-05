#
# Copyright 2017 Quantopian, Inc.
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
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from plottable import Table
from scipy import stats

from . import plotting
from . import performance as perf
from . import utils


class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None


@plotting.customize
def create_summary_tear_sheet(
    factor_data, long_short=True, group_neutral=False
):
    """
    Creates a small summary tear sheet with returns, information, and turnover
    analysis.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    """

    # Returns Analysis
    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, demeaned=long_short, group_adjust=group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    periods = utils.get_forward_returns_columns(factor_data.columns)
    periods = list(map(lambda p: pd.Timedelta(p).days, periods))

    fr_cols = len(periods)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_quantile_statistics_table(factor_data)

    plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row(),
    )

    # Information Analysis
    ic = perf.factor_information_coefficient(factor_data)
    plotting.plot_information_table(ic)

    # Turnover Analysis
    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in range(1, int(quantile_factor.max()) + 1)
            ],
            axis=1,
        )
        for p in periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    plt.show()
    gf.close()


@plotting.customize
def create_returns_tear_sheet(
    factor_data, long_short=True, group_neutral=False, by_group=False
):
    """
    Creates a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
        Additionally factor values will be demeaned across the factor universe
        when factor weighting the portfolio for cumulative returns plots
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots
    by_group : bool
        If True, display graphs separately for each group.
    """

    factor_returns = perf.factor_returns(
        factor_data, long_short, group_neutral
    )

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, factor_returns, long_short, group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row(),
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_rateret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    # Compute cumulative returns from daily simple returns, if '1D'
    # returns are provided.
    if "1D" in factor_returns:
        title = (
            "Factor Weighted "
            + ("Group Neutral " if group_neutral else "")
            + ("Long/Short " if long_short else "")
            + "Portfolio Cumulative Return (1D Period)"
        )

        plotting.plot_cumulative_returns(
            factor_returns["1D"], period="1D", title=title, ax=gf.next_row()
        )

        plotting.plot_cumulative_returns_by_quantile(
            mean_quant_ret_bydate["1D"], period="1D", ax=gf.next_row()
        )

    ax_mean_quantile_returns_spread_ts = [
        gf.next_row() for x in range(fr_cols)
    ]
    plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts,
    )

    plt.show()
    gf.close()

    if by_group:
        (
            mean_return_quantile_group,
            mean_return_quantile_group_std_err,
        ) = perf.mean_return_by_quantile(
            factor_data,
            by_date=False,
            by_group=True,
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        mean_quant_rateret_group = mean_return_quantile_group.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_return_quantile_group.columns[0],
        )

        num_groups = len(
            mean_quant_rateret_group.index.get_level_values("group").unique()
        )

        vertical_sections = 1 + (((num_groups - 1) // 2) + 1)
        gf = GridFigure(rows=vertical_sections, cols=2)

        ax_quantile_returns_bar_by_group = [
            gf.next_cell() for _ in range(num_groups)
        ]
        plotting.plot_quantile_returns_bar(
            mean_quant_rateret_group,
            by_group=True,
            ylim_percentiles=(5, 95),
            ax=ax_quantile_returns_bar_by_group,
        )
        plt.show()
        gf.close()


@plotting.customize
def create_information_tear_sheet(
    factor_data, group_neutral=False, by_group=False
):
    """
    Creates a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_neutral : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, display graphs separately for each group.
    """

    ic = perf.factor_information_coefficient(factor_data, group_neutral)

    plotting.plot_information_table(ic)

    columns_wide = 2
    fr_cols = len(ic.columns)
    rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    if not by_group:

        mean_monthly_ic = perf.mean_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=False,
            by_time="M",
        )
        ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
        plotting.plot_monthly_ic_heatmap(
            mean_monthly_ic, ax=ax_monthly_ic_heatmap
        )

    if by_group:
        mean_group_ic = perf.mean_information_coefficient(
            factor_data, group_adjust=group_neutral, by_group=True
        )

        plotting.plot_ic_by_group(mean_group_ic, ax=gf.next_row())

    plt.show()
    gf.close()


@plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None):
    """
    Creates a tear sheet for analyzing the turnover properties of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    """

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True,
        ).to_numpy()  # get_values()
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(
            turnover_periods,
        )

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if quantile_turnover[period].isnull().all().all():
            continue
        plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[period], period=period, ax=gf.next_row()
        )

    for period in autocorrelation:
        if autocorrelation[period].isnull().all():
            continue
        plotting.plot_factor_rank_auto_correlation(
            autocorrelation[period], period=period, ax=gf.next_row()
        )

    plt.show()
    gf.close()


@plotting.customize
def create_weighted_turnover_tear_sheet(factor_data, turnover_periods=None):
    """
    Creates a tear sheet for analyzing the weighted turnover properties of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    """

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True,
        ).to_numpy()  # get_values()
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(
            turnover_periods,
        )

    weighted_turnover = {p: perf.weighted_turnover(factor_data, period=p) for p in turnover_periods }

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if weighted_turnover[period].isnull().all().all():
            continue
        plotting.plot_weighted_turnover(
            weighted_turnover[period], period=period, ax=gf.next_row()
        )

    plt.show()
    gf.close()


@plotting.customize
def create_Fama_Macbeth_tear_sheet(factor_data):
    """
    Creates a tear sheet for analyzing the Fama-Macbeth regression of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    """
    returns, tvalues, alpha_beta = perf.factor_returns_Fama_Macbeth(factor_data, factor_columns=['factor'])

    fr_cols = len(factor_data.columns)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    print("Fama-Macbeth Regression Returns Analysis")
    utils.print_table(alpha_beta.apply(lambda x: x.round(4)))
    plotting.plot_factor_series(returns, name='return')
    plotting.plot_factor_series(tvalues, name='t-value')
    plt.show()
    fig = gf.fig
    gf.close()
    return fig, alpha_beta


@plotting.customize
def create_full_tear_sheet(factor_data,
                           long_short=True,
                           group_neutral=False,
                           by_group=False):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
    group_neutral : bool
        Should this computation happen on a group neutral portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
        - See tears.create_information_tear_sheet for details on how this
        flag affects information analysis
    by_group : bool
        If True, display graphs separately for each group.
    """

    plotting.plot_quantile_statistics_table(factor_data)
    create_returns_tear_sheet(
        factor_data, long_short, group_neutral, by_group, set_context=False
    )
    create_information_tear_sheet(
        factor_data, group_neutral, by_group, set_context=False
    )
    create_turnover_tear_sheet(factor_data, set_context=False)
    # custom analysis
    create_Fama_Macbeth_tear_sheet(factor_data, set_context=False)
    create_weighted_turnover_tear_sheet(factor_data, set_context=False)


@plotting.customize
def export_full_tear_sheet(factor_data, export_path=Path('.'), name=None,
                           long_short=True,
                           group_neutral=False,
                           by_group=False,
                           turnover_periods=None):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
    group_neutral : bool
        Should this computation happen on a group neutral portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
        - See tears.create_information_tear_sheet for details on how this
        flag affects information analysis
    by_group : bool
        If True, display graphs separately for each group.
    """
    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True,).to_numpy()  # get_values()
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(turnover_periods,)

    from matplotlib.backends.backend_pdf import PdfPages

    res = []
    with PdfPages(export_path / f'{name}.pdf') as pdf:
        quantile_stats = plotting.plot_quantile_statistics_table(factor_data, export=True)
        plt.figure(figsize=(14,9))
        Table(quantile_stats.apply(lambda x:x.round(3)))
        pdf.savefig()
        plt.close()

        # return analysis
        factor_returns = perf.factor_returns(factor_data, long_short, group_neutral)

        mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
            factor_data,
            by_group=False,
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        mean_quant_rateret = mean_quant_ret.apply(
            utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        )

        mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
            factor_data,
            by_date=True,
            by_group=False,
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        )

        compstd_quant_daily = std_quant_daily.apply(
            utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
        )

        alpha_beta = perf.factor_alpha_beta(factor_data, factor_returns, long_short, group_neutral)

        mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
            mean_quant_rateret_bydate,
            factor_data["factor_quantile"].max(),
            factor_data["factor_quantile"].min(),
            std_err=compstd_quant_daily,
        )

        returns_table = plotting.plot_returns_table(alpha_beta, mean_quant_rateret, mean_ret_spread_quant, export=True)
        plt.figure(figsize=(14, 9))
        Table(returns_table.apply(lambda x:x.round(3)))
        pdf.savefig()
        plt.close()

        fr_cols = len(factor_returns.columns)
        vertical_sections = 2 + fr_cols
        if '1D' in factor_returns:
            vertical_sections += 2
        gf = GridFigure(rows=vertical_sections, cols=1)

        plotting.plot_returns_table(
            alpha_beta, mean_quant_rateret, mean_ret_spread_quant
        )

        plotting.plot_quantile_returns_bar(
            mean_quant_rateret,
            by_group=False,
            ylim_percentiles=None,
            ax=gf.next_row(),
        )

        plotting.plot_quantile_returns_violin(
            mean_quant_rateret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
        )

        trading_calendar = factor_data.index.levels[0].freq
        if trading_calendar is None:
            trading_calendar = pd.tseries.offsets.BDay()
            warnings.warn(
                "'freq' not set in factor_data index: assuming business day",
                UserWarning,
            )

        # Compute cumulative returns from daily simple returns, if '1D'
        # returns are provided.
        if "1D" in factor_returns:
            title = (
                    "Factor Weighted "
                    + ("Group Neutral " if group_neutral else "")
                    + ("Long/Short " if long_short else "")
                    + "Portfolio Cumulative Return (1D Period)"
            )

            plotting.plot_cumulative_returns(
                factor_returns["1D"], period="1D", title=title, ax=gf.next_row()
            )

            plotting.plot_cumulative_returns_by_quantile(
                mean_quant_ret_bydate["1D"], period="1D", ax=gf.next_row()
            )

        ax_mean_quantile_returns_spread_ts = [
            gf.next_row() for x in range(fr_cols)
        ]
        plotting.plot_mean_quantile_returns_spread_time_series(
            mean_ret_spread_quant,
            std_err=std_spread_quant,
            bandwidth=0.5,
            ax=ax_mean_quantile_returns_spread_ts,
        )

        pdf.savefig()
        gf.close()

        if by_group:
            (
                mean_return_quantile_group,
                mean_return_quantile_group_std_err,
            ) = perf.mean_return_by_quantile(
                factor_data,
                by_date=False,
                by_group=True,
                demeaned=long_short,
                group_adjust=group_neutral,
            )

            mean_quant_rateret_group = mean_return_quantile_group.apply(
                utils.rate_of_return,
                axis=0,
                base_period=mean_return_quantile_group.columns[0],
            )

            num_groups = len(
                mean_quant_rateret_group.index.get_level_values("group").unique()
            )

            vertical_sections = 1 + (((num_groups - 1) // 2) + 1)
            gf = GridFigure(rows=vertical_sections, cols=2)

            ax_quantile_returns_bar_by_group = [
                gf.next_cell() for _ in range(num_groups)
            ]
            plotting.plot_quantile_returns_bar(
                mean_quant_rateret_group,
                by_group=True,
                ylim_percentiles=(5, 95),
                ax=ax_quantile_returns_bar_by_group,
            )
            pdf.savefig()
            gf.close()

        res.append(alpha_beta.melt(ignore_index=False).set_index('variable', append=True))

        # information analysis
        ic = perf.factor_information_coefficient(factor_data, group_neutral)
        ic_summary = plotting.plot_information_table(ic, export=True)
        plt.figure(figsize=(14, 9))
        Table(ic_summary.apply(lambda x:x.round(3)))
        pdf.savefig()
        plt.close()

        columns_wide = 2
        fr_cols = len(ic.columns)
        rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)

        ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
        plotting.plot_ic_ts(ic, ax=ax_ic_ts)

        ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
        plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
        plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

        if not by_group:
            mean_monthly_ic = perf.mean_information_coefficient(
                factor_data,
                group_adjust=group_neutral,
                by_group=False,
                by_time="M",
            )
            ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
            plotting.plot_monthly_ic_heatmap(
                mean_monthly_ic, ax=ax_monthly_ic_heatmap
            )

        else:
            mean_group_ic = perf.mean_information_coefficient(
                factor_data, group_adjust=group_neutral, by_group=True
            )

            plotting.plot_ic_by_group(mean_group_ic, ax=gf.next_row())

        pdf.savefig()
        gf.close()

        res.append(ic_summary[['IC Mean', 'Risk-adjusted IC']].T.melt(ignore_index=False).set_index('variable', append=True))

        # turnover analysis
        quantile_factor = factor_data["factor_quantile"]

        quantile_turnover = {
            p: pd.concat(
                [
                    perf.quantile_turnover(quantile_factor, q, p)
                    for q in quantile_factor.sort_values().unique().tolist()
                ],
                axis=1,
            )
            for p in turnover_periods
        }

        auto_correlation = pd.concat(
            [
                perf.factor_rank_autocorrelation(factor_data, period)
                for period in turnover_periods
            ],
            axis=1,
        )

        turnover_summary, autocorr_summary = plotting.plot_turnover_table(auto_correlation, quantile_turnover, export=True)
        f, ax = plt.subplots(2, 1, figsize=(14, 16))
        Table(turnover_summary.apply(lambda x:x.round(3)), ax=ax[0])
        Table(autocorr_summary.apply(lambda x:x.round(3)), ax=ax[1])
        pdf.savefig()
        plt.close()

        fr_cols = len(turnover_periods)
        vertical_sections = fr_cols + fr_cols
        gf = GridFigure(rows=vertical_sections, cols=1)

        for period in turnover_periods:
            if quantile_turnover[period].isnull().all().all():
                continue
            plotting.plot_top_bottom_quantile_turnover(
                quantile_turnover[period], period=period, ax=gf.next_row()
            )

        for period in auto_correlation:
            if auto_correlation[period].isnull().all():
                continue
            plotting.plot_factor_rank_auto_correlation(
                auto_correlation[period], period=period, ax=gf.next_row()
            )

        pdf.savefig()
        gf.close()

        # custom analysis
        returns, tvalues, alpha_beta_FM = perf.factor_returns_Fama_Macbeth(factor_data, factor_columns=['factor'])
        alpha_beta_FM.rename(columns={'factor': 'value'}, inplace=True)

        plt.figure(figsize=(14, 9))
        Table(alpha_beta_FM.apply(lambda x:x.round(3)))
        pdf.savefig()
        plt.close()

        print('Fama-Macbeth Regression Returns Analysis')
        utils.print_table(alpha_beta.apply(lambda x:x.round(4)))
        gf = GridFigure(rows=2, cols=1)
        plotting.plot_factor_series(returns, name='return', ax=[gf.next_row()])
        plotting.plot_factor_series(tvalues, name='t-value', ax=[gf.next_row()])
        pdf.savefig()
        gf.close()
        res.append(alpha_beta_FM)

        # weighted turnover
        weighted_turnover = {p: perf.weighted_turnover(factor_data, period=p) for p in turnover_periods}

        fr_cols = len(turnover_periods)
        vertical_sections = fr_cols
        gf = GridFigure(rows=vertical_sections, cols=1)

        for period in turnover_periods:
            if weighted_turnover[period].isnull().all().all():
                continue
            plotting.plot_weighted_turnover(weighted_turnover[period], period=period, ax=gf.next_row())

        pdf.savefig()
        gf.close()

    return res

@plotting.customize
def create_event_returns_tear_sheet(factor_data,
                                    returns,
                                    avgretplot=(5, 15),
                                    long_short=True,
                                    group_neutral=False,
                                    std_bar=True,
                                    by_group=False):
    """
    Creates a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, the factor
        quantile/bin that factor value belongs to and (optionally) the group
        the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after)
        If not None, plot quantile average cumulative returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so then
        factor returns will be demeaned across the factor universe
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile
    by_group : bool
        If True, display graphs separately for each group.
    """

    before, after = avgretplot

    avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        periods_before=before,
        periods_after=after,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    num_quantiles = int(factor_data["factor_quantile"].max())

    vertical_sections = 1
    if std_bar:
        vertical_sections += ((num_quantiles - 1) // 2) + 1
    cols = 2 if num_quantiles != 1 else 1
    gf = GridFigure(rows=vertical_sections, cols=cols)
    plotting.plot_quantile_average_cumulative_return(
        avg_cumulative_returns,
        by_quantile=False,
        std_bar=False,
        ax=gf.next_row(),
    )
    if std_bar:
        ax_avg_cumulative_returns_by_q = [
            gf.next_cell() for _ in range(num_quantiles)
        ]
        plotting.plot_quantile_average_cumulative_return(
            avg_cumulative_returns,
            by_quantile=True,
            std_bar=True,
            ax=ax_avg_cumulative_returns_by_q,
        )

    plt.show()
    gf.close()

    if by_group:
        groups = factor_data["group"].unique()
        num_groups = len(groups)
        vertical_sections = ((num_groups - 1) // 2) + 1
        gf = GridFigure(rows=vertical_sections, cols=2)

        avg_cumret_by_group = perf.average_cumulative_return_by_quantile(
            factor_data,
            returns,
            periods_before=before,
            periods_after=after,
            demeaned=long_short,
            group_adjust=group_neutral,
            by_group=True,
        )

        for group, avg_cumret in avg_cumret_by_group.groupby(level="group"):
            avg_cumret.index = avg_cumret.index.droplevel("group")
            plotting.plot_quantile_average_cumulative_return(
                avg_cumret,
                by_quantile=False,
                std_bar=False,
                title=group,
                ax=gf.next_cell(),
            )

        plt.show()
        gf.close()


@plotting.customize
def create_event_study_tear_sheet(factor_data,
                                  returns,
                                  avgretplot=(5, 15),
                                  rate_of_ret=True,
                                  n_bars=50):
    """
    Creates an event study tear sheet for analysis of a specific event.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
    returns : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).
    rate_of_ret : bool, optional
        Display rate of return instead of simple return in 'Mean Period Wise
        Return By Factor Quantile' and 'Period Wise Return By Factor Quantile'
        plots
    n_bars : int, optional
        Number of bars in event distribution plot
    """

    long_short = False

    plotting.plot_quantile_statistics_table(factor_data)

    gf = GridFigure(rows=1, cols=1)
    plotting.plot_events_distribution(
        events=factor_data["factor"], num_bars=n_bars, ax=gf.next_row()
    )
    plt.show()
    gf.close()

    if returns is not None and avgretplot is not None:

        create_event_returns_tear_sheet(
            factor_data=factor_data,
            returns=returns,
            avgretplot=avgretplot,
            long_short=long_short,
            group_neutral=False,
            std_bar=True,
            by_group=False,
        )

    factor_returns = perf.factor_returns(
        factor_data, demeaned=False, equal_weight=True
    )

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret = mean_quant_ret.apply(
            utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data, by_date=True, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 1
    gf = GridFigure(rows=vertical_sections + 1, cols=1)

    plotting.plot_quantile_returns_bar(
        mean_quant_ret, by_group=False, ylim_percentiles=None, ax=gf.next_row()
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_ret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    plt.show()
    gf.close()


@plotting.customize
def export_event_study_tear_sheet(factor_data,
                                  returns, export_path=Path('.'), name='test',
                                  avgretplot=(5, 15),
                                  rate_of_ret=True,
                                  n_bars=50, group_nuetral=False, std_bar=True, by_group=False):
    """
    Creates an event study tear sheet for analysis of a specific event.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
    returns : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).
    rate_of_ret : bool, optional
        Display rate of return instead of simple return in 'Mean Period Wise
        Return By Factor Quantile' and 'Period Wise Return By Factor Quantile'
        plots
    n_bars : int, optional
        Number of bars in event distribution plot
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile
    by_group : bool
        If True, display graphs separately for each group.
    """
    long_short = False
    res = []
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(export_path / f'{name}.pdf') as pdf:
        quantile_stats = plotting.plot_quantile_statistics_table(factor_data)
        plt.figure(figsize=(14, 8))
        Table(quantile_stats.apply(lambda x:x.round(3)))
        pdf.savefig()
        plt.close()
        res.append(quantile_stats)

        gf = GridFigure(rows=2, cols=1)
        plotting.plot_events_distribution(
            events=factor_data["factor"], num_bars=n_bars, ax=gf.next_row()
        )
        pdf.savefig()
        gf.close()

        if returns is not None and avgretplot is not None:
            before, after = avgretplot
            avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
                factor_data,
                returns,
                periods_before=before,
                periods_after=after,
                demeaned=long_short,
                group_adjust=False,
            )

            num_quantiles = int(factor_data["factor_quantile"].max())

            vertical_sections = 1
            if std_bar:
                vertical_sections += ((num_quantiles - 1) // 2) + 1
            cols = 2 if num_quantiles != 1 else 1
            gf = GridFigure(rows=vertical_sections, cols=cols)
            plotting.plot_quantile_average_cumulative_return(
                avg_cumulative_returns,
                by_quantile=False,
                std_bar=False,
                ax=gf.next_row(),
            )
            if std_bar:
                ax_avg_cumulative_returns_by_q = [
                    gf.next_cell() for _ in range(num_quantiles)
                ]
                plotting.plot_quantile_average_cumulative_return(
                    avg_cumulative_returns,
                    by_quantile=True,
                    std_bar=True,
                    ax=ax_avg_cumulative_returns_by_q,
                )

            pdf.savefig()
            gf.close()

            if by_group:
                groups = factor_data["group"].unique()
                num_groups = len(groups)
                vertical_sections = ((num_groups - 1) // 2) + 1
                gf = GridFigure(rows=vertical_sections, cols=2)

                avg_cumret_by_group = perf.average_cumulative_return_by_quantile(
                    factor_data,
                    returns,
                    periods_before=before,
                    periods_after=after,
                    demeaned=long_short,
                    group_adjust=group_nuetral,
                    by_group=True,
                )

                for group, avg_cumret in avg_cumret_by_group.groupby(level="group"):
                    avg_cumret.index = avg_cumret.index.droplevel("group")
                    plotting.plot_quantile_average_cumulative_return(
                        avg_cumret,
                        by_quantile=False,
                        std_bar=False,
                        title=group,
                        ax=gf.next_cell(),
                    )

                pdf.savefig()
                gf.close()

        factor_returns = perf.factor_returns(
            factor_data, demeaned=False, equal_weight=True
        )

        mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
            factor_data, by_group=False, demeaned=long_short
        )
        res.append(mean_quant_ret)

        if rate_of_ret:
            mean_quant_ret = mean_quant_ret.apply(
                utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
            )

        mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
            factor_data, by_date=True, by_group=False, demeaned=long_short
        )
        if rate_of_ret:
            mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
                utils.rate_of_return,
                axis=0,
                base_period=mean_quant_ret_bydate.columns[0],
            )

        fr_cols = len(factor_returns.columns)
        vertical_sections = 2 + fr_cols * 1
        gf = GridFigure(rows=vertical_sections + 1, cols=1)

        plotting.plot_quantile_returns_bar(
            mean_quant_ret, by_group=False, ylim_percentiles=None, ax=gf.next_row()
        )

        plotting.plot_quantile_returns_violin(
            mean_quant_ret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
        )

        trading_calendar = factor_data.index.levels[0].freq
        if trading_calendar is None:
            trading_calendar = pd.tseries.offsets.BDay()
            warnings.warn(
                "'freq' not set in factor_data index: assuming business day",
                UserWarning,
            )

        pdf.savefig()
        gf.close()
    return res


@plotting.customize
def export_ts_tear_sheet(factor_data, returns,
                         export_path=Path('.'), name='test', thres_value=None, thres_quantile=None,
                         avgretplot=(5, 15), rate_of_ret=True,
                         n_bars=50, group_nuetral=False, std_bar=True, by_group=False):
    """
    Creates a full tear sheet for analysis and evaluationg single time series.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
    returns : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).
    rate_of_ret : bool, optional
        Display rate of return instead of simple return in 'Mean Period Wise
        Return By Factor Quantile' and 'Period Wise Return By Factor Quantile'
        plots
    n_bars : int, optional
        Number of bars in event distribution plot
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile
    by_group : bool
        If True, display graphs separately for each group.
    """
    long_short = False
    res = {}
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(export_path / f'{name}.pdf') as pdf:
        # factor distribution description
        plt.figure(figsize=(14, 8))
        factor_data['factor'].hist(bins=20)
        pdf.savefig()
        plt.close()

        # quantile stats
        if 'factor_quantile' in factor_data.columns and len(factor_data['factor_quantile'].unique()) > 1:
            quantile_stats = plotting.plot_quantile_statistics_table(factor_data)
            plt.figure(figsize=(14, 9))
            Table(quantile_stats.apply(lambda x: x.round(3)))
            pdf.savefig()
            plt.close()
            res['quantile_stat'] = quantile_stats

        # time series correlation between factor and returns
        forward_ret_cols = utils.get_forward_returns_columns(factor_data)
        ic = {col: stats.spearman(factor_data[col], factor_data['factor'])[0] for col in forward_ret_cols}
        ic_pearson = {col:stats.pearsonr(factor_data[col], factor_data['factor'])[0] for col in forward_ret_cols}
        ic = pd.DataFrame([ic, ic_pearson], index=['ic', 'ic_pearson'])
        plt.figure(figsize=(14, 9))
        Table(ic.apply(lambda x:x.round(4)))
        pdf.savefig()
        plt.close()

        res['ic'] = ic

        # returns analysis group by quantiles/bins
        group_stats = factor_data.groupby('factor_quantile')[forward_ret_cols].agg(['mean', 'std', 'count'])
        for col in forward_ret_cols:
            tmp_stat = group_stats.loc[:, col].copy()
            tmp_stat['t-value'] = tmp_stat['mean'] / tmp_stat['std']
            tmp_stat.columns = [x+'_'+col for x in tmp_stat.columns]
            plt.figure(figsize=(14,9))
            Table(tmp_stat.apply(lambda x: x.round(6)))
            pdf.savefig()
            plt.close()

        res['quantile_return'] = group_stats

        # factor return analysis using event study if thres_value/quantile is not None
        if thres_value is not None or thres_quantile is not None:
            if thres_value is not None:
                event_data = factor_data[factor_data['factor'].abs() > thres_value]
                event_data.loc[event_data['factor'] < -thres_value, 'factor_quantile'] = 1
                event_data.loc[event_data['factor'] > thres_value, 'factor_quantile'] = 2
            else:
                thres_quantile = sorted([thres_quantile, 1-thres_quantile])
                quantile_thres = factor_data['factor'].quantile(thres_quantile)
                event_data = factor_data[(factor_data['factor'] < quantile_thres.iloc[0]) |
                                         (factor_data['factor'] > quantile_thres.iloc[1])]
                event_data.loc[event_data['factor'] < quantile_thres.iloc[0], 'factor_quantile'] = 1
                event_data.loc[event_data['factor'] > quantile_thres.iloc[1], 'factor_quantile'] = 2

            # distribution
            gf = GridFigure(rows=2, cols=1)
            plotting.plot_events_distribution(
                events=event_data["factor"], num_bars=n_bars, ax=gf.next_row()
            )
            pdf.savefig()
            gf.close()

            # information
            ic_event = {col: stats.spearman(event_data[col], event_data['factor'])[0] for col in forward_ret_cols}
            ic_pearson_event = {col: stats.pearsonr(event_data[col], event_data['factor'])[0] for col in forward_ret_cols}
            ic_event = pd.DataFrame([ic_event, ic_pearson_event], index=['ic', 'ic_pearson'])
            plt.figure(figsize=(14, 9))
            Table(ic_event.apply(lambda x: x.round(4)))
            pdf.savefig()
            plt.close()
            res['ic_event'] = ic_event

            if returns is not None and avgretplot is not None:
                before, after = avgretplot
                avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
                    event_data,
                    returns,
                    periods_before=before,
                    periods_after=after,
                    demeaned=long_short,
                    group_adjust=False,
                )

                avg_cumulative_returns.sort_index(axis=1, inplace=True)
                avg_cumulative_returns.loc[(slice(None), 'mean'), :] = (avg_cumulative_returns.loc[(slice(None), 'mean'), :].values /
                                                                        avg_cumulative_returns.loc[(slice(None), 'mean'), 0].values)


                num_quantiles = int(event_data["factor_quantile"].max())

                vertical_sections = 1
                vertical_sections += ((num_quantiles - 1) // 2) + 1
                cols = 2 if num_quantiles != 1 else 1
                gf = GridFigure(rows=vertical_sections, cols=cols)
                plotting.plot_quantile_average_cumulative_return(
                    avg_cumulative_returns,
                    by_quantile=False,
                    std_bar=False,
                    ax=gf.next_row(),
                )
                ax_avg_cumulative_returns_by_q = [
                    gf.next_cell() for _ in range(num_quantiles)
                ]
                plotting.plot_quantile_average_cumulative_return(
                    avg_cumulative_returns,
                    by_quantile=True,
                    std_bar=True,
                    ax=ax_avg_cumulative_returns_by_q,
                )

                pdf.savefig()
                gf.close()

            event_returns = perf.factor_returns(
                event_data, demeaned=False, equal_weight=True
            )

            mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
                event_data, by_group=False, demeaned=long_short
            )

            if rate_of_ret:
                mean_quant_ret = mean_quant_ret.apply(
                    utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
                )

            mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
                event_data, by_date=True, by_group=False, demeaned=long_short
            )
            if rate_of_ret:
                mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
                    utils.rate_of_return,
                    axis=0,
                    base_period=mean_quant_ret_bydate.columns[0],
                )

            fr_cols = len(event_returns.columns)
            vertical_sections = 2 + fr_cols * 1
            gf = GridFigure(rows=vertical_sections + 1, cols=1)

            plotting.plot_quantile_returns_bar(
                mean_quant_ret, by_group=False, ylim_percentiles=None, ax=gf.next_row()
            )

            plotting.plot_quantile_returns_violin(
                mean_quant_ret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
            )

            pdf.savefig()
            gf.close()
    return res