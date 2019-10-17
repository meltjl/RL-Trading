from env.StockEnv import StockEnv
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
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

from stable_baselines import A2C, ACKTR, DQN, DDPG, SAC, PPO1, PPO2, TD3, TRPO
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
lr = 0.01
learningRates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
set_global_seeds(seed)
np.random.seed(seed)


def dateparse1(x): return pd.datetime.strptime(x, '%Y%m%d')


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=lr, n_steps=1, gamma=0.7, env=e),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e, learning_rate=5e-4, n_steps=1),
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, lam=0.5, optim_batchsize=16, optim_stepsize=1e-3),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e, learning_rate=lr, lam=0.8),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e, max_kl=0.05, lam=0.7),
    'ddpg': lambda e: DDPG(policy="MlpPolicy", env=e, gamma=0.1, buffer_size=int(1e6)),
}


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


'''
policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy,
    'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
model = PPO2(policy=policy, env=env, n_steps=n_steps, nminibatches=nminibatches,
    lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
    learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
                 '''


def TrainWith_BackTest(algo, df, model_name, portfolio_name):

    # ref https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    split = 4
    splits = TimeSeriesSplit(n_splits=split)
    before = np.zeros(split)
    after = np.zeros(split)
    backtest = np.zeros(split)
    train_dates = np.empty(split, dtype="datetime64[s]")
    test_dates = np.empty(split, dtype="datetime64[s]")
    timestamp = datetime.now().strftime("%Y%m%d %H%M%s")

    # for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    # for s in np.random.randint(0,5):
    # for lr in [1e-1, 1e-3, 1e-5]:
    # for lr in [1e-1, 1e-2]:
    loop = 0

    for train_index, test_index in splits.split(df.values):
        print("loop", loop)
        train = df.iloc[train_index, ]
        test = df.iloc[test_index, ]
        timestamp = datetime.now().strftime("%Y%m%d %H%M%s")
        logdir = "./log/"
        logfile = logdir + portfolio_name
        uniqueTrainId = model_name + "_" + timestamp + "_Train" + str(loop)
        train_dates[loop] = max(train.date)
        test_dates[loop] = max(test.date)

        # choose environment
        # env = StockTradingEnv(train_df)

        # add noise
        # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(
            n_actions), sigma=0.1 * np.ones(n_actions))
        global env
        env = DummyVecEnv(
            [lambda: StockEnv(train, logfile + "_" + uniqueTrainId + ".csv", model_name)])

        # Automatically normalize the input features
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

        # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
        model = DDPG("MlpPolicy", env, gamma=0.1, buffer_size=int(1e6))
        #model = PPO2(MlpPolicy, env, verbose=1, learning_rate=lr)
        #model = algo(MlpPolicy, env, verbose=1)

        # Random Agent, before training
        print("*** Agent before learning ***")
        steps = len(np.unique(train.date))
        before[loop] = evaluate(model, num_steps=steps)

        print("*** Set agent to learn ***")
        model.learn(total_timesteps=round(steps))

        print("*** Evaluate the trained agent ***")
        after[loop] = evaluate(model, num_steps=steps)

        # Save the agent
        # model.save("model/" + uniqueId)

        # delete trained model to demonstrate loading. This also frees up memory
        # del model

        # load model ##### NEED TO UPDATE THIS TO BECOME !!!!!!
        # model = PPO2.load("model/" + model_name + timestamp)

        print("*** Run agent on unseen data ***")
        uniqueTestId = model_name + "_" + timestamp + "_BackTest" + str(loop)
        env = DummyVecEnv(
            [lambda: StockEnv(test, logfile + "_" + uniqueTestId + ".csv", uniqueTestId)])

        # Automatically normalize the input features
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        steps = len(np.unique(test.date))
        backtest[loop] = evaluate(model, num_steps=steps)

        loop += 1
    #algo_idx += 1

    for i in range(split):
        print("\ntrain_dates:", min(df.date), train_dates[i])
        print("backtest {} : before | after | backtest : {: 8.2f} | {: 8.2f} | {: 8.2f}".format(
            i, before[i], after[i], backtest[i]))

    data = pd.DataFrame({"timestamp": timestamp, "Model": uniqueTrainId,  "Seed": seed, "learningRate": lr,
                         "backtest  # ": np.arange(split), "StartTrainDate": min(train.date),
                         "EndTrainDate": train_dates, "before": before,
                         "after": after, "testDate": test_dates, "roadTest": backtest})
    data.head()
    with open('summary.csv', 'a') as f:
        data.to_csv(f, header=True)
        # data.to_csv(f, header=False)


def TrainSingle(config):
    model_name = "ppo2"
    refreshData = 0
    portfolio = 0

    df = get_data(config, portfolio=portfolio, refreshData=refreshData)
    print(df.head())
    print(df.info())

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

    env = DummyVecEnv([lambda: StockEnv(train, logfile, model_name, seed=seed)])
    # env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # model = PPO2(MlpPolicy, env, verbose=1)
    # model = LEARN_FUNC_DICT[model_name](env)
    # https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py
    # model = DDPG("MlpPolicy", env, gamma=0.1, action_noise=action_noise, buffer_size=int(1e6))

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


def chkArgs(argv):
    try:
        opts, args = getopt.getopt(
            argv, "hm:o:p:r:", ["model=PPO2", "portfolio=", "refreshData=True", "ofile="])
    except getopt.GetoptError:
        print('main.py')
        sys.exit(2)

    model_name = "ppo2"
    refreshData = 0
    portfolio = 0

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

    portfolio_name = config["portfolios"][portfolio]["name"]

    model_name = "PPO2"
    algo = PPO2
    TrainWith_BackTest(algo, df, model_name, portfolio_name)

    model_name = "DDPG"
    algo = DDPG
    TrainWith_BackTest(algo, df, model_name, portfolio_name)

    model_name = "ACKTR"
    algo = ACKTR
    #TrainWith_BackTest(algo, df, model_name, portfolio_name)

    model_name = "A2C"
    algo = A2C
    #TrainWith_BackTest(algo, df, model_name, portfolio_name)


if __name__ == "__main__":
    chkArgs(sys.argv[1:])
