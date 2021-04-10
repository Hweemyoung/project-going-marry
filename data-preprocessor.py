import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self,
                 chart,
                 header=['date', 'open', 'high', 'low', 'close', 'volume', 'kospi', 'vol_ant', 'vol_for', 'vol_ins',
                         'vol_srt', 'vol_pow'],
                 us_treasury_10=None,
                 gold=None,
                 silver=None,
                 copper=None,
                 paladium=None,
                 dubai=None,
                 brent=None,
                 log_base=1.02,
                 windows=[5, 20, 60, 120]):
        self.windows = windows
        self.chart = chart
        self.header = header
        self.gold = gold
        self.silver = silver
        self.copper = copper
        self.paladium = paladium
        self.dubai = dubai
        self.brent = brent
        self.log_base = log_base
        self.windows = windows

        self.ma_columns = list()
        self.target_columns = list()

    def __add_features(self, data):
        # Ratios
        _base = np.log(self.log_base)
        
        # 장외 격동성: log(당일 시가/전일 종가)
        data['open_lastclose_ratio'] = np.zeros(len(data))
        data.loc[1:, 'open_lastclose_ratio'] = np.log(data['open'][1:].values / data['close'][:-1].values) / _base
        self.ma_columns.append('open_lastclose_ratio')
        
        # 장내 격동성: log(당일 고가/당일 저가)
        data['high_low_ratio'] = np.log(data['high'].values / data['low'].values) / _base
        self.ma_columns.append('high_low_ratio')
        
        # 주가 변화율: log(당일 종가/전일 종가)
        data['close_lastclose_ratio'] = np.zeros(len(data))
        data.loc[1:, 'close_lastclose_ratio'] = np.log(data['close'][1:].values / data['close'][:-1].values) / _base
        self.ma_columns.append('close_lastclose_ratio')
        
        # 거래량 변화율: log(당일 거래량/전일 거래량)
        data['vol_lastvol_ratio'] = np.zeros(len(data))
        data.loc[1:, 'vol_lastvol_ratio'] = np.log(data['vol'][1:].values / data['vol'][:-1].values) / _base
            # .replace(to_replace=0, method='ffill') \
            # .replace(to_replace=0, method='bfill').values
        self.ma_columns.append('vol_lastvol_ratio')

        return data

    def __ma(self, data, windows):
        for _col in data.columns:
            for _window in windows:
                # Calculate MA
                _col_temp = '%f{_col}_ma%f{_window}'
                data[_col_temp] = data[_col].rolling(_window).mean()
                # Calculate MA ratio
                _col_new = '%f{_col_temp}_ratio'
                self.target_columns.append(_col_new)
                data[_col_new] = (
                    data[_col] - data[_col_temp]) / data[_col_temp]

        return data


COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V1_RICH = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'inst_lastinst_ratio', 'frgn_lastfrgn_ratio',
    'inst_ma5_ratio', 'frgn_ma5_ratio',
    'inst_ma10_ratio', 'frgn_ma10_ratio',
    'inst_ma20_ratio', 'frgn_ma20_ratio',
    'inst_ma60_ratio', 'frgn_ma60_ratio',
    'inst_ma120_ratio', 'frgn_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V2 = [
    'per', 'pbr', 'roe',
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio',
    'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio',
    'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio',
    'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio'
]


def preprocess(data, ver='v1'):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data['close_ma{}'.format(window)] = \
            data['close'].rolling(window).mean()
        data['volume_ma{}'.format(window)] = \
            data['volume'].rolling(window).mean()
        data['close_ma%d_ratio' % window] = \
            (data['close'] - data['close_ma%d' % window]) \
            / data['close_ma%d' % window]
        data['volume_ma%d_ratio' % window] = \
            (data['volume'] - data['volume_ma%d' % window]) \
            / data['volume_ma%d' % window]

        if ver == 'v1.rich':
            data['inst_ma{}'.format(window)] = \
                data['close'].rolling(window).mean()
            data['frgn_ma{}'.format(window)] = \
                data['volume'].rolling(window).mean()
            data['inst_ma%d_ratio' % window] = \
                (data['close'] - data['inst_ma%d' % window]) \
                / data['inst_ma%d' % window]
            data['frgn_ma%d_ratio' % window] = \
                (data['volume'] - data['frgn_ma%d' % window]) \
                / data['frgn_ma%d' % window]

    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) \
        / data['close'][:-1].values
    data['high_close_ratio'] = \
        (data['high'].values - data['close'].values) \
        / data['close'].values
    data['low_close_ratio'] = \
        (data['low'].values - data['close'].values) \
        / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) \
        / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = \
        (data['volume'][1:].values - data['volume'][:-1].values) \
        / data['volume'][:-1] \
        .replace(to_replace=0, method='ffill') \
        .replace(to_replace=0, method='bfill').values

    if ver == 'v1.rich':
        data['inst_lastinst_ratio'] = np.zeros(len(data))
        data.loc[1:, 'inst_lastinst_ratio'] = \
            (data['inst'][1:].values - data['inst'][:-1].values) \
            / data['inst'][:-1] \
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values
        data['frgn_lastfrgn_ratio'] = np.zeros(len(data))
        data.loc[1:, 'frgn_lastfrgn_ratio'] = \
            (data['frgn'][1:].values - data['frgn'][:-1].values) \
            / data['frgn'][:-1] \
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    return data


def load_data(fpath, date_from, date_to, ver='v2'):
    header = None if ver == 'v1' else 0
    data = pd.read_csv(fpath, thousands=',', header=header,
                       converters={'date': lambda x: str(x)})

    if ver == 'v1':
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    data = data.sort_values(by='date').reset_index()

    # 데이터 전처리
    data = preprocess(data)

    # 기간 필터링
    data['date'] = data['date'].str.replace('-', '')
    data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
    data = data.dropna()

    # 차트 데이터 분리
    chart_data = data[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = data[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.rich':
        training_data = data[COLUMNS_TRAINING_DATA_V1_RICH]
    elif ver == 'v2':
        data.loc[:, ['per', 'pbr', 'roe']] = \
            data[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception('Invalid version.')

    return chart_data, training_data
