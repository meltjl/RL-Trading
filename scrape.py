import os
import math
import datetime
import getopt
import sys
import pandas as pd
from datetime import datetime
from dateutil import parser
import numpy as np


import json
with open('./config.json', 'r') as f:
    config = json.load(f)

quandl.ApiConfig.api_key = config["api"]

# if not os.path.exists(data_path) or re_download:
print('Start to download market data')
colummns = ['ticker', 'date', 'adj_open', 'adj_close', 'adj_high', 'adj_low', 'adj_volume']
for i in range(len(config["portfolios"]) - 1):
    sample = config["portfolios"][i]
    stocks = quandl.get_table('WIKI/PRICES', ticker=sample["asset"],
                              qopts={'columns': colummns},
                              date={'gte': sample["start_date"], 'lte': sample["end_date"]}, paginate=True)

    stock_groups = stocks.groupby('ticker')
    stocks = {}
    for k in stock_groups.groups.keys():
        group = stock_groups.get_group(k)
        group.index = group.date
        stocks[k] = group[columns[-5:]]
    #market_data = pd.Panel(stocks).fillna(method='ffill').fillna(method='bfill')

    try:
        with open('/data/' + sample["name"]+".csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=colummns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


#dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
print("Path at terminal when executing this file")
print(os.getcwd() + "\n")


columns = ['ticker', 'date', 'adj_open', 'adj_close', 'adj_high', 'adj_low', 'adj_volume']
print(columns[-5:-1])


a = ['asset' + str(i) for i in range(5)]
print(','.join(a))

#print(datetime.datetime.now().strftime("%Y%m%d %h%M%s"))

d = datetime.datetime.now()
print(d.strftime("%Y%m%d %H%M%s"))
regex = ".+(\\.+)$"

total_reward = 100
print(math.exp(total_reward) * 100)

'''
def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('Input file is "', inputfile)
    print('Output file is "', outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])

'''


'''
def dateparse1(x): return pd.datetime.strptime(x, '%Y%m%d')


def dateparse2(x): return pd.datetime.strptime(x, '%Y-%m-%d')


file = '/mnt/share/Code/mel_code/data/mel_MSFT-AAPL.csv'


cutoff_date = '2016-01-01'
df = pd.read_csv(file, parse_dates=['Date'], date_parser=dateparse2)
columns = ['Date', 'Tic', 'Open', 'High', 'Low', 'Close', 'Volume']
df = df.loc[df['Date'] < cutoff_date, columns]
df = df.sort_values(by=['Date', 'Tic'])
# print(df.head())

numSecurity = numSecurity = len(df.Tic.unique())
numTrainDay = len(df.Date.unique())

train_daily_data = []

for date in np.unique(df.Date):
    train_daily_data.append(df[df.Date == date])

x = df[df.Date == '1998-01-02']


state = [100] + x.Close.values.tolist() + [0 for i in range(2)]
print(state)

asset_prices = np.array(state[1:(numSecurity+1)])
asset_quantity = np.array(state[(numSecurity+1):])

end_total_asset = state[0] + sum(asset_prices * asset_quantity)
print(end_total_asset)
'''
