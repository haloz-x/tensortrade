import pandas as pd

df = pd.read_csv("data/ETH-USDT-SWAP.csv", index_col=0, parse_dates=True)


ohlc = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}

# df2 = df.resample('15min', base=15).apply(ohlc)
#
#
# print(df2.head(300))
x = {k: "LTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}

print(x)

