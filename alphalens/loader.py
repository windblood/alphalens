"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-14 22:39:48
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-14 22:56:04
FilePath: 
Description: 因子读取
"""
from abc import ABC, abstractmethod
from typing import List, Union
from functools import reduce, lru_cache
from pathlib import Path

import pandas as pd

import config


def datetime2str(watch_dt: pd.Timestamp, fmt: str = "%Y-%m-%d") -> str:
    return pd.to_datetime(watch_dt).strftime(fmt)


class CSVLoader:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.price_path = data_path / 'MarketData' / 'Day_trade'
        self.factor_path = data_path / 'Factor'

    def get_single_factor(self, file, start_dt, end_dt):
        df = pd.read_hdf(file)
        df = df.loc[start_dt:end_dt]
        df = df.melt(var_name='code', value_name=file.name.split('.')[0], ignore_index=False)
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index.astype(str))
        df = df.set_index('code', append=True)
        return df

    def get_factor_data(self, factor_names: List[str], codes: Union[List, str] = None,
                        start_dt: str = None, end_dt: str = None, ) -> pd.DataFrame:
        if start_dt is None or end_dt is None:
            raise ValueError("start_dt 和 end_dt 不能为空")
        start_dt = datetime2str(start_dt, "%Y%m%d")
        end_dt = datetime2str(end_dt, "%Y%m%d")

        factor_folders = {factor: self.factor_files[factor]
                          for factor in factor_names if factor in self.factor_files}

        df = [self.get_single_factor(f, start_dt, end_dt) for factor, f in factor_folders.items()]
        df = pd.concat(df, axis=1)
        if codes is not None:
            if isinstance(codes, str):
                codes = [codes]
            df = df[df.index.get_level_values(1).isin(codes)]
        df.sort_values('date', inplace=True)
        return df

    def get_single_stock(self, file, start_dt, end_dt):
        df = pd.read_csv(file)
        df = df[['ts_code', 'trade_date', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq', 'vol', 'amount']]
        df = df[(int(start_dt)<=df['trade_date']) * (df['trade_date']<=int(end_dt))]
        return df

    def get_stock_price(self, codes: Union[str, List] = None,
                        start_dt: str = None, end_dt: str = None,) -> pd.DataFrame:
        if start_dt is None or end_dt is None:
            raise ValueError("start_dt 和 end_dt 不能为空")

        start_dt: str = datetime2str(start_dt, "%Y%m%d")
        end_dt: str = datetime2str(end_dt, "%Y%m%d")
        if codes is None:
            files = ([x for x in (self.price_path / 'SSE').iterdir() if x.is_file()] +
                     [x for x in (self.price_path / 'SZE').iterdir() if x.is_file()])
        else:
            files = [self.price_path / ('SSE' if code[0]=='6' else 'SZE') / f'{code[:6]}.csv'
                     for code in codes if code[0] in ['6', '0', '3']]

        df = [self.get_single_stock(f, start_dt, end_dt) for f in files]
        df = pd.concat(df)
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
        df = df.sort_values('trade_date').rename(columns={'ts_code': 'code', 'trade_date': 'date', 'open_hfq': 'open',
                                                          'high_hfq': 'high', 'low_hfq': 'low', 'close_hfq': 'close'})
        df['code'] = df['code'].apply(lambda x: x[:6])
        return df

    @property
    @lru_cache()
    def factor_files(self):
        type_folder = self.factor_path.iterdir()
        factors = [x for folder in type_folder
                   for x in folder.iterdir() if x.is_file()]
        files = {factor.name.split('.')[0]: factor for factor in factors}
        return files

    @property
    def factor_name_list(self) -> List[str]:
        return list(self.factor_files.keys())

    @lru_cache()
    def get_stock_industry(self, start_dt: str = None, end_dt: str = None, standard='zx'):
        folder = 'ZXClass' if standard=='zx' else 'SWClass2'
        class_path = self.data_path / 'MarketInfo' / 'Daily' / folder
        class_comps = [pd.read_hdf(x).assign(code=cls.name) for cls in class_path.iterdir()
                       for x in cls.iterdir() if start_dt <= x.name.split('.')[0] <= end_dt]
        class_comps = pd.concat(class_comps).drop(columns='SecuAbbr')
        class_comps.rename(columns={'EndDate': 'date', 'SecuCode': 'code', 'code': 'group',}, inplace=True)
        # class_comps.set_index(['date', 'code'], inplace=True)
        class_comps.drop_duplicates(subset=['date', 'code'], inplace=True)
        return class_comps

    def get_industry_weight(self, start_dt: str = None, end_dt: str = None, benchmark='HS300', standard='zx'):
        start_dt: str = datetime2str(start_dt, "%Y%m%d")
        end_dt: str = datetime2str(end_dt, "%Y%m%d")
        weight_path = self.data_path / 'MarketInfo' / 'Daily' / 'Weight' / benchmark
        all_dates = [x for x in weight_path.iterdir()
                     if start_dt<=x.name.split('.')[0]<=end_dt]
        index_weights = [pd.read_hdf(x) for x in all_dates]
        index_weights = pd.concat(index_weights)
        index_weights.rename(columns={'EndDate': 'date', 'SecuCode': 'code', 'WeightRatio': 'weight'}, inplace=True)
        # index_weights.set_index(['date', 'code'], inplace=True)

        class_comps = self.get_stock_industry(start_dt=start_dt, end_dt=end_dt, standard=standard)
        df = pd.merge(index_weights, class_comps, on=['date', 'code'], how='inner')
        df = df.groupby(['date', 'group'])['weight'].sum().reset_index()
        df.drop_duplicates(subset=['date', 'group'], inplace=True)
        return df


class MysqlLoader:
    def __init__(self, host: str, port: int, username: str, password: str, db: str) -> None:
        import pymysql
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.db_name = db

        # 连接数据库
        self.db = pymysql.connect(host=self.host, port=self.port,
                                  user=self.username, password=self.password, database=self.db_name)
        self.cursor = self.db.cursor()

    def __del__(self) -> None:
        try:
            self.cursor.close()
            self.db.close()
        except Exception:
            pass

    def get_factor_data(self, factor_names: List[str], codes: Union[List, str] = None,
                        start_dt: str = None, end_dt: str = None, ) -> pd.DataFrame:
        if start_dt is None or end_dt is None:
            raise ValueError("start_dt 和 end_dt 不能为空")
        start_dt = datetime2str(start_dt, "%Y%m%d")
        end_dt = datetime2str(end_dt, "%Y%m%d")
        sel_time_expr = f"date >= {start_dt} and date <= {end_dt}"

        if codes is not None:
            if isinstance(codes, str):
                sel_code_expr = f"ts_code == '{codes}'"
            elif isinstance(codes, (list, set, tuple)):
                codes_str = '","'.join(codes)
                sel_code_expr = f'ts_code in ("{codes_str}")'
            else:
                sel_code_expr = ""

        df = []
        for factor_name in factor_names:
            expr = f" select * from  {factor_name} where {sel_time_expr}"
            if codes is not None:
                expr += f" and {sel_code_expr}"

            self.cursor.execute(expr)
            tmp_df = self.cursor.fetchall()
            tmp_df = pd.DataFrame(tmp_df, columns=['trade_date', 'code', 'value'])
            tmp_df['factor_name'] = factor_name
            df.append(tmp_df.drop_duplicates())
        df = pd.concat(df, axis=0)
        df = df.sort_values('trade_date')
        return df

    @property
    def factor_name_list(self) -> List[str]:
        expr = "show tables from factor"
        self.cursor.execute(expr)
        tables = [x[0] for x in self.cursor.fetchall()]  # 'Tables_in_factor'
        factor_tables = []
        for tbl in tables:
            tmp_expr = f"select * from `{tbl}` limit 1"
            self.cursor.execute(tmp_expr)
            tmp_col = [x[0] for x in self.cursor.description]
            if len(tmp_col) == 3 and 'value' in tmp_col:
                factor_tables.append(tbl)
        return factor_tables

    def get_stock_price(self, codes: Union[str, List] = None,
                        start_dt: str = None, end_dt: str = None,) -> pd.DataFrame:
        if start_dt is None or end_dt is None:
            raise ValueError("start_dt 和 end_dt 不能为空")

        start_dt: str = datetime2str(start_dt, "%Y%m%d")
        end_dt: str = datetime2str(end_dt, "%Y%m%d")
        sel_time_expr = f"trade_date >= {start_dt} and trade_date <= {end_dt}"
        cols = 'ts_code,trade_date,open_hfq,high_hfq,low_hfq,close_hfq,vol,amount'
        expr = f" select {cols} from  stockquote where {sel_time_expr}"

        if codes is not None:
            if isinstance(codes, str):
                sel_code_expr = f"ts_code == '{codes}'"
            elif isinstance(codes, (list, set, tuple)):
                codes_str = '","'.join(codes)
                sel_code_expr = f'ts_code in ("{codes_str}")'
            else:
                sel_code_expr = ""
            expr += f" and {sel_code_expr}"

        self.cursor.execute(expr)
        df = self.cursor.fetchall()
        df = pd.DataFrame(df, columns=cols.split(','))
        df = df.sort_values('trade_date').rename(columns={'ts_code': 'code', 'open_hfq': 'open', 'high_hfq': 'high',
                                                          'low_hfq': 'low', 'close_hfq': 'close'})
        df['code'] = df['code'].apply(lambda x: x[:6])
        return df

    def get_stock_industry(self, codes: Union[str, List] = None,
                           start_dt: str = None, end_dt: str = None, standard='zx'):
        if start_dt is None or end_dt is None:
            raise ValueError("start_dt 和 end_dt 不能为空")

        start_dt: str = datetime2str(start_dt, "%Y-%m-%d %H:%M:%S")
        end_dt: str = datetime2str(end_dt, "%Y-%m-%d %H:%M:%S")
        sel_time_expr = f"EndDate >= '{start_dt}' and EndDate <= '{end_dt}'"
        tbl = 'zxclass' if standard == 'zx' else 'swclass2'
        expr = f" select * from  {tbl} where {sel_time_expr}"

        if codes is not None:
            if isinstance(codes, str):
                sel_code_expr = f"SecuCode == '{codes}'"
            elif isinstance(codes, (list, set, tuple)):
                codes_str = '","'.join(codes)
                sel_code_expr = f'SecuCode in ("{codes_str}")'
            else:
                sel_code_expr = ""
            expr += f" and {sel_code_expr}"

        self.cursor.execute(expr)
        df = self.cursor.fetchall()
        df = pd.DataFrame(df, columns=['date', 'code', 'name', 'group'])
        df = df.sort_values('date')
        self.stock_industry = df
        return df

    def get_industry_weight(self, start_dt: str = None, end_dt: str = None, benchmark='HS300'):
        if start_dt is None or end_dt is None:
            raise ValueError("start_dt 和 end_dt 不能为空")

        start_dt: str = datetime2str(start_dt, "%Y-%m-%d %H:%M:%S")
        end_dt: str = datetime2str(end_dt, "%Y-%m-%d %H:%M:%S")
        sel_time_expr = f"EndDate >= '{start_dt}' and EndDate <= '{end_dt}'"
        tbl = 'indexweight'
        expr = f'select * from  {tbl} where {sel_time_expr} and code="{benchmark}"'

        self.cursor.execute(expr)
        df = self.cursor.fetchall()
        df = pd.DataFrame(df, columns=['date', 'code', 'name', 'weight', 'benchmark'])
        df = df.sort_values('date')
        if 'stock_industry' in dir(self):
            stock_industry = self.stock_industry
        else:
            stock_industry = self.get_stock_industry(start_dt=start_dt, end_dt=end_dt)
        df = pd.merge(df, stock_industry, on=['date', 'code', 'name'], how='left')
        res = df.groupby(['date', 'group'])['weight'].sum() / 100
        return res


class Loader(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_factor_data(
            self, factor_name: Union[List, str], start_dt: str, end_dt: str
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_stock_price(
            self,
            codes: Union[str, List],
            start_dt: str,
            end_dt: str,
            fields: Union[str, List],
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def factor_name_list(self) -> List[str]:
        pass

    @abstractmethod
    def get_factor_begin_and_end_period(self, factor_name: str) -> List[str]:
        pass


# class DolphindbLoader(Loader):
#     def __init__(
#             self,
#             host: str = config.DB_CONN["host"],
#             port: int = config.DB_CONN["port"],
#             username: str = config.DB_CONN["username"],
#             password: str = config.DB_CONN["password"],
#     ) -> None:
#         import dolphindb as ddb
#         # 连接数据库
#         self.session: ddb.session = ddb.session()
#         self.session.connect(host, port, username, password)
#         self.factor_data: pd.DataFrame = None
#         self.stock_price: pd.DataFrame = None
#
#     def get_factor_data(
#             self, factor_name: Union[List, str], start_dt: str, end_dt: str
#     ) -> pd.DataFrame:
#         if not isinstance(factor_name, (str, list)):
#             raise ValueError("factor_name must be str or list")
#
#         sel_factor: str = (
#             f"factor_name=='{factor_name}'"
#             if isinstance(factor_name, str)
#             else f"factor_name in {factor_name}"
#         )
#
#         time_between: List[str] = []
#         if start_dt is not None:
#             start_dt_str: str = datetime2str(start_dt, "%Y.%m.%d")
#             time_between.append(f"trade_date >= {start_dt_str}")
#         if end_dt is not None:
#             end_dt_str: str = datetime2str(end_dt, "%Y.%m.%d")
#             time_between.append(f"trade_date <= {end_dt_str}")
#         time_between_str: str = " and ".join(time_between)
#
#         expr: str = (
#             f"{sel_factor} and ({time_between_str})" if time_between_str else sel_factor
#         )
#
#         query_expr: str = f"""
#         factor_table = loadTable('{config.FACTPR_DB_PATH}', '{config.FACTOR_TABLE_NAME}')
#         select * from factor_table where {expr} and (code like '6%SH' or code like '3%SZ' or code like '0%SZ')
#         """
#
#         self.factor_data = self.session.run(query_expr, clearMemory=True)
#         return self.factor_data
#
#     def get_stock_price(
#             self,
#             codes: Union[str, List],
#             start_dt: str,
#             end_dt: str,
#             fields: Union[str, List],
#     ) -> pd.DataFrame:
#         fields: List[str] = [fields] if isinstance(fields, str) else fields
#
#         default_fields: List[str] = ["code", "trade_date"] + [
#             field for field in fields if field not in ["trade_date", "code"]
#         ]
#         default_fields_str: str = ",".join(default_fields)
#
#         if not isinstance(codes, (str, list)):
#             raise ValueError("codes must be str or list")
#
#         sel_codes: str = {str: f"code=='{codes}'", list: f"code in {codes}"}.get(
#             type(codes), ""
#         )
#
#         time_between: List[str] = []
#         if start_dt is not None:
#             start_dt_str: str = datetime2str(start_dt, "%Y.%m.%d")
#             time_between.append(f"trade_date >= {start_dt_str}")
#         if end_dt is not None:
#             end_dt_str: str = datetime2str(end_dt, "%Y.%m.%d")
#             time_between.append(f"trade_date <= {end_dt_str}")
#         time_between_str: str = " and ".join(time_between)
#
#         expr: str = (
#             f"{sel_codes} and ({time_between_str})" if time_between else sel_codes
#         )
#
#         query_expr: str = f"""
#         price_table = loadTable('{config.PRICE_DB_PATH}', '{config.PRICE_TABLE_NAME}')
#         select {default_fields_str} from price_table where {expr} and (code like '6%SH' or code like '3%SZ' or code like '0%SZ')
#         """
#
#         self.stock_price = self.session.run(query_expr, clearMemory=True)
#         return self.stock_price
#
#     @property
#     def factor_name_list(self) -> List[str]:
#         expr = f"""
#         table = loadTable('{config.FACTPR_DB_PATH}', '{config.FACTOR_TABLE_NAME}')
#         schema(table).partitionSchema[1]
#         """
#
#         factor_name: List[str] = self.session.run(expr, clearMemory=True).tolist()
#         return [factor for factor in factor_name if factor not in ["f1", "f2"]]
#
#     def get_factor_begin_and_end_period(self, factor_name: str) -> List[str]:
#         if not isinstance(factor_name, str):
#             raise ValueError("factor_name must be str")
#
#         query_expr: str = f"""
#         factor_table = loadTable('{config.FACTPR_DB_PATH}', '{config.FACTOR_TABLE_NAME}')
#         select min(trade_date),max(trade_date) from factor_table where factor_name == "{factor_name}"
#         """
#         return (
#             self.session.run(query_expr, clearMemory=True)
#             .iloc[0]
#             .dt.strftime("%Y-%m-%d")
#             .tolist()
#         )
#

if __name__ == '__main__':
    loader = MysqlLoader(host='localhost', port='3306', username='wingtrade',
                         password='wingtrade123', db='factor')
    factor_list = loader.factor_name_list
    factor_data = loader.get_factor_data(factor_names=factor_list[:2], start_dt='20190101', end_dt='20231231')
    factor_data['trade_date'] = pd.to_datetime(factor_data['trade_date'].astype(str))
    factor_name = factor_list[0]
    factor_ser = (factor_data.set_index(["trade_date", "code"])
        .query("factor_name==@factor_name")["value"]
        .sort_index()
        .dropna())
    codes = [x for x in factor_data["code"].unique().tolist() if x is not None]
    ind_df = loader.get_stock_industry(codes=codes, start_dt='20190101', end_dt='20240217')
    ind_dict = dict(zip(ind_df['code'], ind_df['group']))
    full_codes = [x + '.SH' if x.startswith('6') else x + '.SZ' for x in codes]
    stock_price = loader.get_stock_price(codes=full_codes, start_dt='20190101', end_dt='20231231')
    stock_price['trade_date'] = pd.to_datetime(stock_price['trade_date'].astype(str))
    pricing: pd.DataFrame = pd.pivot_table(
        stock_price, index="trade_date", columns="code", values='close'
    )

    quantiles = 5