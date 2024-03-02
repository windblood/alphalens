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
from functools import reduce
import pandas as pd


def datetime2str(watch_dt: pd.Timestamp, fmt: str = "%Y-%m-%d") -> str:
    return pd.to_datetime(watch_dt).strftime(fmt)


class CSVLoader:
    def __init__(self, price_path: str, factor_path: str) -> None:
        self.price_path = price_path
        self.factor_path = factor_path

    def get_factor_data(
        self,
        codes: Union[List, str] = None,
        start_dt: str = None,
        end_dt: str = None,
        fields: Union[str, List] = None,
    ) -> pd.DataFrame:
        if isinstance(fields, str):
            fields: List[str] = [fields]
        fields: List[str] = list({"trade_date", "code"}.union(fields))
        df: pd.DataFrame = pd.read_csv(self.factor_path, parse_dates=True)
        df.sort_values("trade_date", inplace=True)

        df: pd.DataFrame = df.query(
            "trade_date >= @start_dt and trade_date <= @end_dt"
        )[fields]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if codes:
            df: pd.DataFrame = df[df["code"].isin(codes)]

        return df

    def get_stock_price(
        self,
        codes: Union[str, List],
        start_dt: str,
        end_dt: str,
        fields: Union[str, List],
    ) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(self.price_path, parse_dates=True)
        df.sort_values("trade_date", inplace=True)
        fields: List[str] = list({"trade_date", "code"}.union(fields))
        df: pd.DataFrame = df.query(
            "trade_date >= @start_dt and trade_date <= @end_dt"
        )[fields]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if codes:
            df: pd.DataFrame = df[df["code"].isin(codes)]

        return df

    def get_factor_name(self) -> List[str]:
        # 获取csv文件的列名
        df: pd.DataFrame = pd.read_csv(
            self.factor_path, index_col=None, parse_dates=True, nrows=1
        )
        return [col for col in df.columns if col not in ["code", "trade_date"]]

    @property
    def get_factor_name_list(self) -> List[str]:
        factor_name: List[str] = ['ROE', 'PB', 'PE', 'MOM', 'beta']
        return [factor for factor in factor_name if factor not in ["f1", "f2"]]


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
    def get_factor_name_list(self) -> List[str]:
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
        df = pd.DataFrame(df, columns=['date', 'code', 'name', 'ind_code'])
        df = df.sort_values('date')
        return df

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
    def get_factor_name_list(self) -> List[str]:
        pass

    @abstractmethod
    def get_factor_begin_and_end_period(self, factor_name: str) -> List[str]:
        pass


if __name__ == '__main__':
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
    full_codes = [x + '.SH' if x.startswith('6') else x + '.SZ' for x in codes]
    stock_price = loader.get_stock_price(codes=full_codes, start_dt='20190101', end_dt='20231231')
    stock_price['trade_date'] = pd.to_datetime(stock_price['trade_date'].astype(str))
    pricing: pd.DataFrame = pd.pivot_table(
        stock_price, index="trade_date", columns="code", values='close'
    )

    quantiles = 5