from env.StockEnv import StockEnv
import gym
import numpy as np
import pandas as pd
import sys
import os
import csv
import json
import getopt
import quandl
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, SAC, PPO1, PPO2, TD3, TRPO
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.bench import Monitor

# This is sample trading env from
# paper   : 01B) https://towardsdatascience.com/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4
# code :    01B) https://github.com/notadamking/Stock-Trading-Visualization
# but performance is really bad. Makes 38% losses
# from env.StockTradingEnv import StockTradingEnv


# this is from xiong's code. renamed zxStock_env into StockEnv but use stable_baselines concept as per 01B paper

# tf.set_random_seed(42)
seed = 42
lr = 1e-3
set_global_seeds(seed)
np.random.seed(seed)


def dateparse1(x): return pd.datetime.strptime(x, '%Y%m%d')


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=lr, n_steps=1, gamma=0.7, env=e).learn(total_timesteps=10000, seed=seed),
    'acer': lambda e: ACER(policy="MlpPolicy", env=e, n_steps=1, replay_ratio=1).learn(total_timesteps=15000, seed=seed),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e, learning_rate=5e-4, n_steps=1).learn(total_timesteps=20000, seed=seed),
    'dqn': lambda e: DQN(policy="MlpPolicy", batch_size=16, gamma=0.1, exploration_fraction=0.001, env=e).learn(total_timesteps=40000, seed=seed),
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, lam=0.5,
                           optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=15000, seed=seed),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e, learning_rate=1.5e-3, lam=0.8).learn(total_timesteps=20000, seed=seed),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e, max_kl=0.05, lam=0.7).learn(total_timesteps=10000, seed=seed),
}


def dataPrep(file, cutoff_date, dateparse):
    # xiong data prep
    # simple split data into train & test by date
    df = pd.read_csv(file, parse_dates=['Date'], date_parser=dateparse)
    columns = ['Date', 'Tic', 'Open', 'High', 'Low', 'Close', 'Volume']
    train = df.loc[df['Date'] < cutoff_date, columns]
    test = df.loc[df['Date'] >= cutoff_date, columns]

    # sort by date and tic
    train = train.sort_values(by=['Date', 'Tic'])
    test = test.sort_values(by=['Date', 'Tic'])

    numSecurity = len(train.Tic.unique())
    print("Number of security in TRAIN")
    print(numSecurity, train.Tic.unique())
    print("Train shape | Min | Max date : ", train.shape, "\t|",
          train.Date.min(), "|", train.Date.max())

    print("\nNumber of security in TEST")
    print(numSecurity, test.Tic.unique())
    print("Test shape  | Min | Max date : ", test.shape, "\t|",
          test.Date.min(), "|", test.Date.max())

    # !!! TO DO
    # detrend / normalize !!!!!!!
    return train, test


def train(train, env):
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: env])
    print('observation_space :\t', env.observation_space)
    print('action_space :\t', env.action_space)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50)

    obs = env.reset()

    # write columsn header
    filename = 'render.txt'
    file = open(filename, 'w+')
    file.write('step\tbalance\tshares_held\ttotal_shares_sold\tcost_basis\t\
               total_sales_value\tnet_worth\tmax_net_worth\tprofit')
    file.close()

    # for i in range(10):
    # env.render()
    episode_rewards = [0.0]
    episodes = len(np.unique(train['date']))
    for i in range(episodes):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # 20190927 - temporary disable. need to fix render in StockEnv.py
        # env.render(title="MSFT")
        # env.render(title="MSFT", mode='file', filename=filename)
        env.render()
        # Stats
        episode_rewards[-1] += rewards
        if done:
            obs = env.reset()

    episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = np.mean(episode_rewards[-100:])
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))


def evaluate(model, num_steps=1000):
    episode_rewards = [0.0]
    obs = env.reset()
    print("\nIn Evaluate")
    print('observation_space :\t', env.observation_space)
    print('action_space :\t', env.action_space)

    # env.render()

    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # 20190927 - temporary disable. need to fix render in StockEnv.py
        # env.render(title="MSFT")
        # env.render(title="MSFT", mode='file', filename=filename)
        # env.render()

        # Stats
        episode_rewards[-1] += rewards
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    # Compute mean reward for the last 100 episodes
    mean_reward = np.mean(episode_rewards)
    print("Mean reward:", mean_reward)
    return mean_reward


def get_data(config, portfolio=0, refreshData=False):
    columns = ['ticker', 'date', 'adj_open', 'adj_close', 'adj_high', 'adj_low', 'adj_volume']
    sample = config["portfolios"][portfolio]
    file = "./data/" + sample["name"] + ".csv"

    if not os.path.exists(file) or refreshData:
        print('Start to download market data')
        quandl.ApiConfig.api_key = config["api"]
        df = quandl.get_table('WIKI/PRICES', ticker=sample["asset"], qopts={'columns': columns}, date={
            'gte': sample["start_date"], 'lte': sample["end_date"]}, paginate=True)

        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.sort_values(by=["date", "ticker"])
        df.to_csv(file)
        print(file, "saved")
    else:
        print('Loading file', file)
        df = pd.read_csv(file, parse_dates=['date'], date_parser=dateparse).fillna(
            method='ffill').fillna(method='bfill')
        df = df.sort_values(by=["date", "ticker"])
    return df


def pre_process(df, open_c, high_c, low_c, close_c, volume_c):
    preprocessed_data = {}
    cleaned_data = {}
    for c in market_data.items:
        columns = [open_c, close_c, high_c, low_c, volume_c]
        security = df[c, :, columns].fillna(method='ffill').fillna(method='bfill')
        security[volume_c] = security[volume_c].replace(0, np.nan).fillna(method='ffill')
        cleaned_data[c] = security.copy()
        tech_data = _get_indicators(security=security.astype(
            float), open_name=open_c, close_name=close_c, high_name=high_c, low_name=low_c, volume_name=volume_c)
        preprocessed_data[c] = tech_data
    preprocessed_data = pd.Panel(preprocessed_data).dropna()
    cleaned_data = pd.Panel(cleaned_data)[:, preprocessed_data.major_axis, :].dropna()
    return preprocessed_data, cleaned_data


def _get_indicators(security, open_name, close_name, high_name, low_name, volume_name):
    open_price = security[open_name].values
    close_price = security[close_name].values
    low_price = security[low_name].values
    high_price = security[high_name].values
    volume = security[volume_name].values if volume_name else None
    security['MOM'] = talib.MOM(close_price)
    security['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)
    security['HT_DCPHASE'] = talib.HT_DCPHASE(close_price)
    security['SINE'], security['LEADSINE'] = talib.HT_SINE(close_price)
    security['INPHASE'], security['QUADRATURE'] = talib.HT_PHASOR(close_price)
    security['ADXR'] = talib.ADXR(high_price, low_price, close_price)
    security['APO'] = talib.APO(close_price)
    security['AROON_UP'], _ = talib.AROON(high_price, low_price)
    security['CCI'] = talib.CCI(high_price, low_price, close_price)
    security['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price)
    security['PPO'] = talib.PPO(close_price)
    security['MACD'], security['MACD_SIG'], security['MACD_HIST'] = talib.MACD(close_price)
    security['CMO'] = talib.CMO(close_price)
    security['ROCP'] = talib.ROCP(close_price)
    security['FASTK'], security['FASTD'] = talib.STOCHF(high_price, low_price, close_price)
    security['TRIX'] = talib.TRIX(close_price)
    security['ULTOSC'] = talib.ULTOSC(high_price, low_price, close_price)
    security['WILLR'] = talib.WILLR(high_price, low_price, close_price)
    security['NATR'] = talib.NATR(high_price, low_price, close_price)
    security['RSI'] = talib.RSI(close_price)
    security['EMA'] = talib.EMA(close_price)
    security['SAREXT'] = talib.SAREXT(high_price, low_price)
    # security['TEMA'] = talib.EMA(close_price)
    security['RR'] = security[close_name] / security[close_name].shift(1).fillna(1)
    security['LOG_RR'] = np.log(security['RR'])
    if volume_name:
        security['MFI'] = talib.MFI(high_price, low_price, close_price, volume)
        # security['AD'] = talib.AD(high_price, low_price, close_price, volume)
        # security['OBV'] = talib.OBV(close_price, volume)
        security[volume_name] = np.log(security[volume_name])
    security.drop([open_name, close_name, high_name, low_name], axis=1)
    security = security.dropna().astype(np.float32)
    return security


def main_bak(argv):
    inputfile = ''
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('main.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('Input file is "', inputfile)

    # assume file open is successful
    inputfile = 'data/mel_DJ.csv'
    # train_df, test_df = dataPrep(inputfile, '2016-01-01', dateparse1)

    # losses-38%
    inputfile = 'data/mel_MSFT-AAPL.csv'
    train_df, test_df = dataPrep(inputfile, '1998-01-10', dateparse2)

    # train_df, test_df = dataPrep(inputfile, '2016-01-10', dateparse2)
    # choose environment
    # env = StockTradingEnv(train_df)
    env = StockEnv(train_df)

    train(train_df, env)


def main2(argv):

    try:
        opts, args = getopt.getopt(argv, "hm:o:", ["model=", "ofile="])
    except getopt.GetoptError:
        print('main.py -m <model> ')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('main.py -m <model> -o <ofile>')
            sys.exit()
        elif opt in ("-o", "--ofile"):
            model_name = arg
        elif opt in ("-m", "--mfile"):
            model_name = arg

    with open('./config.json', 'r') as f:
        config = json.load(f)

    df = get_data(config, portfolio=1, refreshData=False)

    # temp hard code model
    # model_name = 'ppo2'

    # assume file open is successful
    inputfile = 'data/mel_DJ.csv'
    timestamp = datetime.now().strftime("%Y%m%d %H%M%s")
    logfile = 'log/mel_DJ_log_' + model_name + '_' + timestamp + '.csv'
    # train_df, test_df = dataPrep(inputfile, '2016-01-01', dateparse1)

    inputfile = 'data/mel_MSFT-AAPL.csv'
    logfile = 'log/mel_MSFT-AAPL_log_' + model_name + '_' + timestamp + '.csv'
    # train_df, test_df = dataPrep(inputfile, '1998-01-31', dateparse2)
    train_df, test_df = dataPrep(inputfile, '2018-01-10', dateparse2)

    # choose environment
    # env = StockTradingEnv(train_df)

    # add noise
    # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # The algorithms require a vectorized environment to run
    global env
    set_global_seeds(0)

    env = DummyVecEnv([lambda: StockEnv(train_df, logfile, model_name)])
    # model = PPO2(MlpPolicy, env, verbose=1)
    model = LEARN_FUNC_DICT[model_name](env)
    # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
    # model = DDPG("MlpPolicy", env, gamma=0.1, action_noise=action_noise, buffer_size=int(1e6))

    # Random Agent, before training
    steps = len(np.unique(train_df['Date']))
    # before = evaluate(model, num_steps=10)

    # let model learn
    print("*** Set agent loose to learn ***")
    model.learn(total_timesteps=round(steps))

    # Save the agent
    # model.save("model/" + model_name + timestamp)

    # delete trained model to demonstrate loading. This also frees up memory
    # del model

    # load model ##### NEED TO UPDATE THIS TO BECOME !!!!!!
    # model = PPO2.load("model/" + model_name + timestamp)

    print("*** Evaluate the trained agent ***")
    after = evaluate(model, num_steps=steps)
    # print("before | after:", before, after)


'''
policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
model = PPO2(policy=policy, env=env, n_steps=n_steps, nminibatches=nminibatches,
    lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
    learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
                 '''


def mainSplit(argv):
    try:
        opts, args = getopt.getopt(
            argv, "hm:o:p:r:", ["model=PPO2", "portfolio=", "refreshData=True", "ofile="])
    except getopt.GetoptError:
        print('main.py')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('python main.py -m ppo2 -o <ofile>')
            sys.exit()
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-m", "--mfile"):
            model_name = arg
        elif opt in ("-p", "--portfolio"):
            portfolio = int(arg)
        elif opt in ("-r", "--refreshData"):
            refreshData = arg

    with open('./config.json', 'r') as f:
        config = json.load(f)

    model_name = "ppo2"
    refreshData = 0
    portfolio = 0
    #cutoff_date = '2017-01-01'

    df = get_data(config, portfolio=portfolio, refreshData=refreshData)
    print(df.head())
    print(df.info())

    # The algorithms require a vectorized environment to run
    global env

    # ref https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    split = 4
    splits = TimeSeriesSplit(n_splits=split)
    loop = 0
    before = np.zeros(split)
    after = np.zeros(split)
    train_dates = np.empty(split, dtype="datetime64[s]")
    test_dates = np.empty(split, dtype="datetime64[s]")

    for train_index, test_index in splits.split(df.values):
        print("loop", loop)

        # print("\ntrain | test index", train_index, test_index)
        train = df.iloc[train_index, ]
        test = df.iloc[test_index, ]
        timestamp = datetime.now().strftime("%Y%m%d %H%M%s")
        logdir = "./log/"
        logfile = logdir + config["portfolios"][portfolio]["name"] + \
            "_" + model_name + "_" + timestamp + ".csv"

        train_dates[loop] = max(train.date)
        # test_dates[loop] = (min(test.date), max(test.date))

        print("train: start | end | no Tickers |", min(train.date),
              max(train.date), "|", len(train.ticker.unique()))
        print("test : start | end | no Tickers |", min(test.date),
              max(test.date), "|", len(test.ticker.unique()))

        print(train_dates[loop])
        train_step = len(train_index)
        # test_step = len(test_step)
        # print(train_step, test_step)

        # choose environment
        # env = StockTradingEnv(train_df)

        # add noise
        # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        env = DummyVecEnv([lambda: StockEnv(train, logfile, model_name)])

        # Automatically normalize the input features
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

        # model = PPO2(MlpPolicy, env, verbose=1)
        # model = LEARN_FUNC_DICT[model_name](env)
        # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
        # model = DDPG("MlpPolicy", env, gamma=0.1, action_noise=action_noise, buffer_size=int(1e6))

        model = PPO2(MlpPolicy, env, verbose=1, learning_rate=0.1)

        # Random Agent, before training
        print("random agent")
        steps = len(np.unique(train.date))
        # steps = len(train_index)
        before[loop] = evaluate(model, num_steps=steps)

        # let model learn
        print("*** Set agent loose to learn ***")
        model.learn(total_timesteps=round(steps))

        # Save the agent
        # model.save("model/" + model_name + timestamp)

        # delete trained model to demonstrate loading. This also frees up memory
        # del model

        # load model ##### NEED TO UPDATE THIS TO BECOME !!!!!!
        # model = PPO2.load("model/" + model_name + timestamp)

        print("*** Evaluate the trained agent ***")
        after[loop] = evaluate(model, num_steps=steps)
        loop += 1

    for i in range(split):
        print("\ntrain_dates:", min(df.date), train_dates[i])
        print("backtest {} : before | after : {: 8.2f} | {: 8.2f}".format(i, before[i], after[i]))


def mainOK(argv):
    try:
        opts, args = getopt.getopt(
            argv, "hm:o:p:r:", ["model=PPO2", "portfolio=", "refreshData=True", "ofile="])
    except getopt.GetoptError:
        print('main.py')
        sys.exit(2)

    model_name = "ppo2"
    refreshData = 0
    portfolio = 0
    cutoff_date = '2017-01-01'

    for opt, arg in opts:
        if opt == '-h':
            print('python main.py -m ppo2 -o <ofile>')
            sys.exit()
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-m", "--mfile"):
            model_name = arg
        elif opt in ("-p", "--portfolio"):
            portfolio = int(arg)
        elif opt in ("-r", "--refreshData"):
            refreshData = arg

    with open('./config.json', 'r') as f:
        config = json.load(f)

    df = get_data(config, portfolio=portfolio, refreshData=refreshData)
    print(df.head())
    print(df.info())

    # print("\ntrain | test index", train_index, test_index)

    train = df.loc[df['date'] < cutoff_date, ]
    test = df.loc[df['date'] >= cutoff_date, ]

    timestamp = datetime.now().strftime("%Y%m%d %H%M%s")
    logfile = "./log/" + config["portfolios"][portfolio]["name"] + \
        "_" + model_name + "_" + timestamp + ".csv"

    # print(train.head())
    # print(test.head())

    # choose environment
    # env = StockTradingEnv(train_df)

    # add noise
    # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
    n_actions = 3
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # The algorithms require a vectorized environment to run
    global env
    set_global_seeds(0)

    env = DummyVecEnv([lambda: StockEnv(train, logfile, model_name, seed=seed)])
    #env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # model = PPO2(MlpPolicy, env, verbose=1)
    # model = LEARN_FUNC_DICT[model_name](env)
    # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
    #model = DDPG("MlpPolicy", env, gamma=0.1, action_noise=action_noise, buffer_size=int(1e6))

    model = PPO2(MlpPolicy, env, verbose=1)

    # Random Agent, before training
    steps = len(np.unique(train.date))
    before = evaluate(model, num_steps=steps)

    # let model learn
    print("*** Set agent loose to learn ***")
    model.learn(total_timesteps=round(steps))
    print("before :", before)

    # Save the agent
    # model.save("model/" + model_name + timestamp)

    # delete trained model to demonstrate loading. This also frees up memory
    # del model

    # load model ##### NEED TO UPDATE THIS TO BECOME !!!!!!
    # model = PPO2.load("model/" + model_name + timestamp)

    print("*** Evaluate the trained agent ***")
    after = evaluate(model, num_steps=steps)
    print("before | after:", before, after)

    print("train: start | end | no Tickers |", min(train.date), "|",
          max(train.date), "|", len(train.ticker.unique()))
    print("test : start | end | no Tickers |", min(test.date), "|",
          max(test.date), "|", len(test.ticker.unique()))


def bak():

        # assume file open is successful
    inputfile = 'data/mel_DJ.csv'
    timestamp = datetime.now().strftime("%Y%m%d %H%M%s")
    logfile = 'log/mel_DJ_log_' + model_name + '_' + timestamp + '.csv'
    # train_df, test_df = dataPrep(inputfile, '2016-01-01', dateparse1)

    inputfile = 'data/mel_MSFT-AAPL.csv'
    logfile = 'log/mel_MSFT-AAPL_log_' + model_name + '_' + timestamp + '.csv'
    # train_df, test_df = dataPrep(inputfile, '1998-01-31', dateparse2)
    train_df, test_df = dataPrep(inputfile, '2018-01-10', dateparse2)

    # choose environment
    # env = StockTradingEnv(train_df)

    # add noise
    # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # The algorithms require a vectorized environment to run
    global env
    set_global_seeds(0)

    env = DummyVecEnv([lambda: StockEnv(train_df, logfile, model_name)])
    # model = PPO2(MlpPolicy, env, verbose=1)
    model = LEARN_FUNC_DICT[model_name](env)
    # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
    # model = DDPG("MlpPolicy", env, gamma=0.1, action_noise=action_noise, buffer_size=int(1e6))

    # Random Agent, before training
    steps = len(np.unique(train_df['Date']))
    # before = evaluate(model, num_steps=10)

    # let model learn
    print("*** Set agent loose to learn ***")
    model.learn(total_timesteps=round(steps))

    # Save the agent
    # model.save("model/" + model_name + timestamp)

    # delete trained model to demonstrate loading. This also frees up memory
    # del model

    # load model ##### NEED TO UPDATE THIS TO BECOME !!!!!!
    # model = PPO2.load("model/" + model_name + timestamp)

    print("*** Evaluate the trained agent ***")
    after = evaluate(model, num_steps=steps)
    # print("before | after:", before, after)


if __name__ == "__main__":
    # mainOK(sys.argv[1:])
    mainSplit(sys.argv[1:])
