from env.StockEnv import StockEnv
from env.StockEnvPlayer import StockEnvPlayer
import gym
import numpy as np
import pandas as pd
import sys
import csv
import os
import csv
import json
import getopt
import quandl
import talib
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing

from stable_baselines import A2C, ACKTR, DQN, DDPG, SAC, PPO1, PPO2, TD3, TRPO
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.ddpg.policies import LnMlpPolicy
# from stable_baselines.bench import Monitor

# This is sample trading env from
# paper   : 01B) https://towardsdatascience.com/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4
# code :    01B) https://github.com/notadamking/Stock-Trading-Visualization
# but performance is really bad. Makes 38% losses
# from env.StockTradingEnv import StockTradingEnv


# this is from xiong's code. renamed zxStock_env into StockEnv but use stable_baselines concept as per 01B paper


# tf.set_random_seed(42)

seed = 42
lr = 1e-2
cliprange = 0.3
g = 0.99

set_global_seeds(seed)
np.random.seed(seed)


def dateparse1(x): return pd.datetime.strptime(x, '%Y%m%d')


def dateparse2(x): return pd.datetime.strptime(x, '%Y/%m/%d')


def dateparse3(x): return pd.datetime.strptime(x, '%b %d, %Y')


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


def evaluate(model, num_steps=1000):
    episode_rewards = [0.0]
    obs = env.reset()
    env.render()

    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

        # Stats
        episode_rewards[-1] += rewards
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    return np.sum(episode_rewards)


def get_data(config, portfolio=0, refreshData=False, addTA='N'):
    columns = ['ticker', 'date', 'adj_open', 'adj_close', 'adj_high', 'adj_low', 'adj_volume']
    sample = config["portfolios"][portfolio]
    file = "./data/" + sample["name"] + ".csv"

    if not os.path.exists(file) or refreshData:
        print('Start to download market data')
        quandl.ApiConfig.api_key = config["api"]
        df = quandl.get_table('WIKI/PRICES', ticker=sample["asset"], qopts={'columns': columns}, date={
            'gte': sample["start_date"], 'lte': sample["end_date"]}, paginate=True)

        df = pre_process(df, addTA='N')
        df.to_csv(file)
        print(file, "saved")
    else:
        print('Loading file', file)
        df = pd.read_csv(file, parse_dates=['date'], date_parser=dateparse)
        df = pre_process(df, addTA)
    return df


def pre_process(df, addTA='N'):
    df = df.sort_values(by=["ticker", "date", ])
    d = df.date.unique()
    tmp = pd.DataFrame({"date": d}, index=d)

    tickers = df.ticker.unique()
    df2 = pd.DataFrame()
    for t in tickers:
        ticker = df.loc[df.ticker == t]
        # force all stock to have same date range
        ticker = pd.merge(tmp, ticker, how='left', on='date')
        ticker.fillna(method='ffill').fillna(method='bfill')

        # add Techical Analysis to each stock
        if addTA == 'Y':
            ticker = add_techicalAnalysis(ticker)
            ticker = ticker.fillna(method='ffill').fillna(method='bfill')

        df2 = pd.concat([df2, ticker], axis=0)
    # df2.to_csv("p3.csv")
    return df2.sort_values(by=["date", "ticker"])


def add_techicalAnalysis(df):
    # open_price = df["adj_open"].values
    close_price = df["adj_close"].values
    # low_price = df["adj_low"].values
    # high_price = df["adj_high"].values
    # volume = df["adj_volume"].values

    #'EMA', 'TEMA',
    #'APO', 'CMO', 'MACD', 'MACD_SIG', 'MACD_HIST', 'MOM', 'PPO', 'ROCP', 'RSI', 'TRIX'
    #'HT_DCPERIOD', 'HT_DCPHASE', 'SINE', 'LEADSINE', 'INPHASE',    'QUADRATURE'

    # =====================================
    # Overlap Studies
    # =====================================
    df['EMA'] = talib.EMA(close_price)
    # TEMA - Triple Exponential Moving Average
    df['TEMA'] = talib.EMA(close_price)
    # WMA - Weighted Moving Average
    #df['WMA'] = talib.WMA(close_price, timeperiod=30)
    # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    #df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close_price)

    # =====================================
    # Momentum Indicator Functions
    # =====================================
    # APO - Absolute Price Oscillator
    df['APO'] = talib.APO(close_price, fastperiod=12, slowperiod=26, matype=0)
    # CMO - Chande Momentum Oscillator
    df['CMO'] = talib.CMO(close_price, timeperiod=14)
    # MACD - Moving Average Convergence/Divergence
    df['MACD'], df['MACD_SIG'], df['MACD_HIST'] = talib.MACD(
        close_price, fastperiod=12, slowperiod=26, signalperiod=9)
    # MOM - Momentum
    df['MOM'] = talib.MOM(close_price)
    # PPO - Percentage Price Oscillator
    df['PPO'] = talib.PPO(close_price, fastperiod=12, slowperiod=26, matype=0)
    # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    df['ROCP'] = talib.ROCP(close_price, timeperiod=10)
    # RSI - Relative Strength Index
    df['RSI'] = talib.RSI(close_price, timeperiod=14)
    # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    df['TRIX'] = talib.TRIX(close_price)

    # NOT USED
    # ADXR - Average Directional Movement Index Rating
    # df['ADXR'] = talib.ADXR(high_price, low_price, close_price)
    # AROON - Aroon
    # df['AROON_UP'], _ = talib.AROON(high_price, low_price)
    # CCI - Commodity Channel Index
    # df['CCI'] = talib.CCI(high_price, low_price, close_price)
    # ULTOSC - Ultimate Oscillator
    # df['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price)
    # WILLR - Williams' %R
    # df['WILLR'] = talib.WILLR(high_price, low_price, close_price)

    # =====================================
    # Cycle Indicator Functions
    # =====================================
    # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
    # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
    # HT_SINE - Hilbert Transform - SineWave
    df['SINE'], df['LEADSINE'] = talib.HT_SINE(close_price)
    # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
    #df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close_price)
    # HT_PHASOR - Hilbert Transform - Phasor Components
    df['INPHASE'], df['QUADRATURE'] = talib.HT_PHASOR(close_price)

    # NOT USED
    # df['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price)
    # df['FASTK'], df['FASTD'] = talib.STOCHF(high_price, low_price, close_price)
    # df['NATR'] = talib.NATR(high_price, low_price, close_price)

    return df


def train(algo, df, model_name, uniqueId, lr=None, gamma=None, noBacktest=1, cutoff_date=None, commission=0, addTA='N'):
    before = np.zeros(noBacktest)
    after = np.zeros(noBacktest)
    backtest = np.zeros(noBacktest)
    train_dates = np.empty(noBacktest, dtype="datetime64[s]")
    start_test_dates = np.empty(noBacktest, dtype="datetime64[s]")
    end_test_dates = np.empty(noBacktest, dtype="datetime64[s]")
    # print(str(df.columns.tolist()))

    dates = np.unique(df.date)
    logfile = "./log/"
    print("noBacktest", noBacktest)

    # backtest=1 uses cut of date to split train/test
    cutoff_date = np.datetime64(cutoff_date)
    print("cutoff_date", cutoff_date)

    if noBacktest == 1:
        a = np.where(dates <= cutoff_date)[0]
        b = np.where(dates > cutoff_date)[0]
        s = []
        s.append((a, b))

    else:
        # ref https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
        splits = TimeSeriesSplit(n_splits=noBacktest)
        s = splits.split(dates)

    loop = 0
    for train_date_index, test_date_index in s:
        print("loop", loop)
        train = df[df.date.isin(dates[train_date_index])]
        test = df[df.date.isin(dates[test_date_index])]
        runtimeId = uniqueId + "_" + str(loop)
        train_dates[loop] = max(train.date)
        start_test_dates[loop] = min(test.date)
        end_test_dates[loop] = max(test.date)

        # normalize
        # train = pd.DataFrame(np.concatenate((train.iloc[:, :3], preprocessing.scale(train.iloc[:, 3:])), axis=1), columns=df.columns)
        # print(train.head())
        # test = pd.DataFrame(np.concatenate((test.iloc[:, :3], preprocessing.scale(test.iloc[:, 3:])), axis=1), columns=df.columns)

        # add noise
        # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(
            n_actions), sigma=0.1 * np.ones(n_actions))
        global env

        # choose environment
        # env = StockTradingEnv(train_df)
        # vectorized environments allow to easily multiprocess training.
        title = runtimeId + "_Train lr=" + \
            str(lr) + ", cliprange=" + str(cliprange) + ", commission=" + str(commission)
        env = DummyVecEnv(
            [lambda: StockEnvPlayer(train, logfile + runtimeId + ".csv", title, seed=seed, commission=commission, addTA=addTA)])

        # Automatically normalize the input features
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

        # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
        # model = DDPG("MlpPolicy", env, gamma=0.1, buffer_size=int(1e6))
        # model = PPO2(MlpPolicy, env, verbose=1, learning_rate=lr)
        # model = algo(MlpPolicy, env, verbose=1, learning_rate=lr, seedy=seed)
        # model = LEARN_FUNC_DICT[model_name](env)

        model = algo(MlpPolicy, env, seedy=seed, gamma=g, n_steps=128,
                     ent_coef=0.01, learning_rate=lr, vf_coef=0.5, max_grad_norm=0.5,
                     lam=0.95, nminibatches=4, noptepochs=4, cliprange=cliprange,
                     cliprange_vf=None,  # tensorboard_log="./tensorlog",
                     _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, )

        # Add some param noise for exploration
        #param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
        # Because we use parameter noise, we should use a MlpPolicy with layer normalization
        #model = DDPG(LnMlpPolicy, env, param_noise=param_noise, verbose=0)

        # Random Agent, before training
        print("*** Agent before learning ***")
        steps = len(np.unique(train.date))
        before[loop] = evaluate(model, num_steps=steps)

        print("*** Set agent to learn ***")
        model.learn(total_timesteps=round(steps))

        print("*** Evaluate the trained agent ***")
        after[loop] = evaluate(model, num_steps=steps)

        # Save the agent
        # model.save("model/" + runtimeId)

        # delete trained model to demonstrate loading. This also frees u memory
        # del model

        # close env
        # env.close()

        # load model - seems like it does not use seed on reloaded model
        # model = algo.load("model/" + runtimeId)

        print("*** Run agent on unseen data ***")
        title = runtimeId + "_Test lr=" + \
            str(lr) + ", cliprange=" + str(cliprange) + ", commission=" + str(commission)
        env = DummyVecEnv(
            [lambda: StockEnvPlayer(test, logfile + runtimeId + ".csv", title, seed=seed, commission=commission, addTA=addTA)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        steps = len(np.unique(test.date))
        backtest[loop] = evaluate(model, num_steps=steps)

        loop += 1

    # display result on screen
    for i in range(noBacktest):
        print("PORTFOLIO", uniqueId)
        print("\ntrain_dates:", min(df.date), train_dates[i])
        print("test_dates:", start_test_dates[i], end_test_dates[i])
        print("backtest {} : SUM reward : before | after | backtest : {: 8.2f} | {: 8.2f} | {: 8.2f}".format(
            i, before[i], after[i], backtest[i]))

    return pd.DataFrame({"Model": uniqueId, "addTA": addTA, "Columns": str(df.columns.tolist()), "commission": commission,
                         "Seed": seed, "cliprange": cliprange, "learningRate": lr, "gamma": g,
                         "backtest  # ": np.arange(noBacktest), "StartTrainDate": min(train.date),
                         "EndTrainDate": train_dates, "before": before,
                         "after": after, "testDate": end_test_dates, "Sum Reward@roadTest": backtest})


def chkArgs(argv):
    try:
        opts, args = getopt.getopt(
            argv, "hb:p:t:r", ["backtest=", "portfolio=", "addtechicalAnalysis=", "refreshData=True"])
    except getopt.GetoptError:
        print('main.py')
        sys.exit(2)

    model_name = "ppo2"
    algo = PPO2
    refreshData = 0
    portfolio = 4
    backtest = 1
    addTA = 'N'
    commission = 0

    for opt, arg in opts:
        if opt == '-h':
            print('python main.py -p <portfolio index> -b <number of back  test> -t <Y|N to add techicalAnalysis')
            sys.exit()
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-b", "--backtest"):
            backtest = int(arg)
        elif opt in ("-p", "--portfolio"):
            portfolio = int(arg)
        elif opt in ("-r", "--refreshData"):
            refreshData = arg
        elif opt in ("-t", "--addtechicalAnalysis"):
            addTA = arg

    with open('./config.json', 'r') as f:
        config = json.load(f)

    df = get_data(config, portfolio=portfolio, refreshData=refreshData, addTA=addTA)
    # print(df.head())
    # print(df.info())
    print("\n\n\n\n\n\n\n\n\n")
    # really bad way to choose TA.
    # if addTA == 'Y':
    # df = df[['date', 'ticker', 'adj_close', 'MOM', 'RSI', 'APO', 'HT_DCPERIOD', 'HT_DCPHASE', 'SINE', 'LEADSINE',
    #     'INPHASE', 'QUADRATURE', 'PPO', 'MACD', 'MACD_SIG', 'MACD_HIST', 'CMO', 'ROCP', 'TRIX', 'EMA', 'TEMA']]
    #df = df[['date', 'ticker', 'adj_close', 'QUADRATURE', 'MACD_HIST', 'RSI']]

    portfolio_name = config["portfolios"][portfolio]["name"]
    commission = config["portfolios"][portfolio]["commission"]

    if "cut_off" in config["portfolios"][portfolio]:
        cutoff_date = config["portfolios"][portfolio]["cut_off"]
    else:
        cutoff_date = ''
        backtest = 4 if backtest == '' else backtest

    '''
    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy,
        'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
    model = PPO2(policy=policy, env=env, n_steps=n_steps, nminibatches=nminibatches,
        lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
        learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
                     '''

    # model_name = "ddpg_0.5"
    # algo = DDPG

    uniqueId = model_name + "_" + portfolio_name + "_" + datetime.now().strftime("%Y%m%d %H%M")

    summary = train(algo, df, model_name, uniqueId, lr=lr,
                    gamma=None, noBacktest=backtest, cutoff_date=cutoff_date, commission=commission, addTA=addTA)

    with open('summary.csv', 'a') as f:
        summary.to_csv(f, header=True)


def testSplit(df):
    '''
    Test to guarantee that split is done on dates instead of row count
    '''
    loop = 0
    split = 2
    splits = TimeSeriesSplit(max_train_size=4025, n_splits=split)
    dates = np.unique(df.date)
    backtest = 1
    # cutoff_date = '2018-03-23T00:00:00.000000000'
    cutoff_date = np.datetime64('2016-01-04')

    if backtest == 1:
        a = np.where(dates < cutoff_date)[0]
        b = np.where(dates >= cutoff_date)[0]
        s = []
        s.append((a, b))
    else:
        s = splits.split(dates)

    for train_date_index, test_date_index in s:
        train = df[df.date.isin(dates[train_date_index])]
        test = df[df.date.isin(dates[test_date_index])]
        print("\ntrain", min(train.date), max(train.date))
        print("test ", min(test.date), max(test.date))


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=lr, n_steps=1, gamma=0.7, env=e),
    # 'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e, learning_rate=5e-4, n_steps=1),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e, learning_rate=lr, n_steps=1),
    # 'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, lam=0.5, optim_batchsize=16, optim_stepsize=1e-3),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", seedy=seed, env=e, learning_rate=1e-2, lam=0.95),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e, max_kl=0.05, lam=0.7),
    'ddpg': lambda e: DDPG(policy="MlpPolicy", env=e, gamma=0.1, buffer_size=int(1e6)),
}


if __name__ == "__main__":
    # file = "./data/xiong.csv"
    # df = pd.read_csv(file, parse_dates=['date'], date_parser=dateparse1).fillna(
    #    method='ffill').fillna(method='bfill')
    # df = df.loc[df.ticker.isin(["002066", "600000", "600962", "000985", "000862"])]
    # df = df.sort_values(by=["date", "ticker"])
    # df.to_csv("./data/portfolio9.csv")
    chkArgs(sys.argv[1:])
