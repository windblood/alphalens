![image](https://media.quantopian.com/logos/open_source/alphalens-logo-03.png){.align-center}

# Alphalens

[![GitHub Actions status](https://github.com/quantopian/alphalens/workflows/CI/badge.svg)](https://github.com/quantopian/alphalens/actions?query=workflow%3ACI+branch%3Amaster)

Alphalens is a Python Library for performance analysis of predictive
(alpha) stock factors. Alphalens works great with the
[Zipline](https://www.zipline.io/) open source backtesting library, and
[Pyfolio](https://github.com/quantopian/pyfolio) which provides
performance and risk analysis of financial portfolios. You can try
Alphalens at [Quantopian](https://www.quantopian.com) \-- a free,
community-centered, hosted platform for researching and testing alpha
ideas. Quantopian also offers a [fully managed service for
professionals](https://factset.quantopian.com) that includes Zipline,
Alphalens, Pyfolio, FactSet data, and more.

The main function of Alphalens is to surface the most relevant
statistics and plots about an alpha factor, including:

-   Returns Analysis
-   Information Coefficient Analysis
-   Turnover Analysis
-   Grouped Analysis

## Getting started

With a signal and pricing data creating a factor \"tear sheet\" is a two
step process:

``` python
import alphalens

# Ingest and format data
factor_data = alphalens.utils.get_clean_factor_and_forward_returns(my_factor, 
                                                                   pricing, 
                                                                   quantiles=5,
                                                                   groupby=ticker_sector,
                                                                   groupby_labels=sector_names)

# Run analysis
alphalens.tears.create_full_tear_sheet(factor_data)
```

## Learn more

Check out the [example
notebooks](https://github.com/quantopian/alphalens/tree/master/alphalens/examples)
for more on how to read and use the factor tear sheet. A good starting
point could be
[this](https://github.com/quantopian/alphalens/tree/master/alphalens/examples/alphalens_tutorial_on_quantopian.ipynb)

## Installation

Install with pip:

    pip install alphalens

Install with conda:

    conda install -c conda-forge alphalens

Install from the master branch of Alphalens repository (development
code):

    pip install git+https://github.com/quantopian/alphalens

Alphalens depends on:

-   [matplotlib](https://github.com/matplotlib/matplotlib)
-   [numpy](https://github.com/numpy/numpy)
-   [pandas](https://github.com/pandas-dev/pandas)
-   [scipy](https://github.com/scipy/scipy)
-   [seaborn](https://github.com/mwaskom/seaborn)
-   [statsmodels](https://github.com/statsmodels/statsmodels)

## Usage

A good way to get started is to run the examples in a [Jupyter
notebook](https://jupyter.org/).

To get set up with an example, you can:

Run a Jupyter notebook server via:

``` bash
jupyter notebook
```

From the notebook list page(usually found at `http://localhost:8888/`),
navigate over to the examples directory, and open any file with a .ipynb
extension.

Execute the code in a notebook cell by clicking on it and hitting
Shift+Enter.

## Questions?

If you find a bug, feel free to open an issue on our [github
tracker](https://github.com/quantopian/alphalens/issues).

## Contribute

If you want to contribute, a great place to start would be the
[help-wanted
issues](https://github.com/quantopian/alphalens/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Credits

-   [Andrew Campbell](https://github.com/a-campbell)
-   [James Christopher](https://github.com/jameschristopher)
-   [Thomas Wiecki](https://github.com/twiecki)
-   [Jonathan Larkin](https://github.com/marketneutral)
-   Jessica Stauth (<jstauth@quantopian.com>)
-   [Taso Petridis](https://github.com/tasopetridis)

For a full list of contributors see the [contributors
page.](https://github.com/quantopian/alphalens/graphs/contributors)

## Example Tear Sheet

Example factor courtesy of [ExtractAlpha](https://extractalpha.com/)

![image](https://github.com/quantopian/alphalens/raw/master/alphalens/examples/table_tear.png)

![image](https://github.com/quantopian/alphalens/raw/master/alphalens/examples/returns_tear.png)

![image](https://github.com/quantopian/alphalens/raw/master/alphalens/examples/ic_tear.png)

![](https://github.com/quantopian/alphalens/raw/master/alphalens/examples/sector_tear.png)

## New functions
- [x] orthogonalization    
- [x] Fama-Macbeth regression factor return    
- [x] return analysis based on Fama-Macbeth regression returns    
- [x] turnover analysis with assets weights    
