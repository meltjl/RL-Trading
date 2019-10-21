import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
import csv
from gym import spaces
from datetime import datetime
import matplotlib.pyplot as plt

iteration = 0


class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, logfile, modelName, initial_investment=10000, seed=7):
        super(StockEnv, self).__init__()

        self.dates = df.date.unique()
        self.numSecurity = len(df.ticker.unique())
        self.numTrainDay = len(self.dates)
        self.terminal = False

        self.initial_investment = initial_investment
        self.logfile = logfile
        self.ledger = []
        self.modelName = modelName
        self._seed(seed)

        df.loc[:, 'qty'] = 0
        self.TA_columns = df.columns[4:-1]
        # print(df.head())
        # print("\n\nta columns")
        # print(self.TA_columns)
        self.pivot = df.pivot(index='date', columns='ticker')
        # add one for initial value
        noStates = len(self.pivot.loc[:, "adj_close":].columns) + 1
        # print("noStates", noStates)

        self.pivot = self.pivot.reset_index()
        self.pivot.insert(1, "initial", pd.Series(self.initial_investment))
        self.pivot.set_index('date')
        # print(self.pivot.head())

        # buy or sell maximum shares
        self.action_space = spaces.Box(
            low=-self.numSecurity, high=self.numSecurity, shape=(self.numSecurity,), dtype=np.int8)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(noStates,))
        print('observation_space :\t', self.observation_space)
        print('action_space :\t', self.action_space)

        self.reset()

        # write column header for the first time
        if 1 == 1:
            with open(logfile, 'a+') as f:
                numSecurity = len(df.ticker.unique())
                ap = ['asset' + str(i) + '_price' for i in range(self.numSecurity)]
                aq = ['asset' + str(i) + '_qty' for i in range(self.numSecurity)]
                others = ['asset' + str(i) + '_' + col for i in range(self.numSecurity)
                          for col in self.TA_columns]
                column = 'model, step, date, cash, portfolio, reward,' + \
                    ','.join(ap) + ',' + ','.join(aq) + ',' + ','.join(others) + '\n'
                f.write(column)

    def reset(self):
        self.asset_memory = [self.initial_investment]
        self.day = 0
        self.reward = 0

        # return as range of one day else it becomes a panda series
        self.data = self.pivot[self.day:self.day+1]

        self.value = self.data.loc[:, ['initial']].values[0].tolist()
        # self.value = self.data.loc[:, ['initial']].to_numpy().flatten()
        # print("self.value", type(self.value), self.value)

        self.price = self.data.loc[:, ['adj_close']].values[0].tolist()
        # self.price = self.data.loc[:, ['adj_close']].to_numpy().flatten()
        # print("self.price", type(self.price), self.price)

        self.qty = self.data.loc[:, ['qty']].values[0].tolist()
        # print("self.qty", type(self.qty), self.qty)

        self.ta = self.data.loc[:, self.TA_columns].values[0].tolist()
        # print("self.ta", type(self.ta), self.ta)
        # [initial money]+[prices for each asset]+[owned shares for each asset]+[techical analysis]
        # self.state = pd.concat([self.value, self.price, self.qty, self.ta], axis=1, sort=False)
        # self.state = np.column_stack([self.value, self.price, self.qty, self.ta])
        # print(type(self.value), type(self.price), type(self.qty), type(self.ta))
        self.state = self.value + self.price + self.qty + self.ta

        # self.state = np.column_stack((self.value, self.price, self.qty, self.ta))
        # print("in else", type(self.value), type(self.price), type(self.qty), type(self.ta))

        # print("state")
        # print(self.state)

        return self.state

    def _sell_stock(self, index, action):
        if self.qty[index] > 0:
            quantity = min(abs(action), self.qty[index])
            self.value += self.price[index] * quantity
            self.qty[index] -= quantity

            # update investment and qty
            self.state = self.value.tolist() + self.price + self.qty + self.ta
        else:
            # print("No asset to sell")
            pass

    def _buy_stock(self, index, action):
        min_quantity = self.state[0] // self.price[index]
        quantity = min(min_quantity, action)
        self.value -= self.price[index] * quantity
        self.qty[index] += quantity
        # print(type(self.value.tolist()), type(self.price), type(self.qty), type(self.ta))
        # update investment and qty
        self.state = self.value.tolist() + self.price + self.qty + self.ta

    def step(self, actions):
        self.terminal = self.day >= (self.numTrainDay-1)

        if self.terminal:
            fig, ax = plt.subplots()
            ax.set_title(self.modelName)
            ax.set_ylabel('Total Asset $')
            ax.set_xlabel('Episode')
            ax.plot(self.asset_memory, color='tomato')
            plt.savefig('image/{}.png'.format(self.modelName))
            plt.close()

            print("**** Summary*****")
            print("Model:\t\t\t", self.modelName.upper())
            print("Number of Assets:\t{:8.0f}".format(self.numSecurity))
            print("Initial Investment :\t{:8.2f}".format(self.initial_investment))

            portfolio_value = self.state[0] + sum(np.array(self.price) * np.array(self.qty))
            print("Portfolio Value:\t{:8.2f}".format(portfolio_value))
            print("% Returns:\t\t{:8.2f}%".format((portfolio_value/self.initial_investment-1)*100))
            print("***************")

            # file = open(self.logfile, 'a+')
            # file.write(','.join(self.ledger))
            # file.write(self.ledger)

            #x = self.ledger + [self.pivot.loc[:, self.TA_columns].values.tolist()]
            # print("zzzzz")
            #print(self.pivot.loc[:, self.TA_columns].head())
            # print(x)
            with open(self.logfile, 'a+') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(self.ledger)

            return self.state, self.reward, self.terminal, {}

        else:
            begin_total_asset = (self.value + sum(np.array(self.price) * np.array(self.qty)))
            # print("begin_total_asset", type(begin_total_asset.tolist()), begin_total_asset[0])

            # actions are predicted by the RL algo to spit out the quantity to buy/sell
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[:: -1][: np.where(actions > 0)[0].shape[0]]
            # print("in else", type(self.value), type(self.price), type(self.qty), type(self.ta))

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            # get next day's price & ta
            self.day += 1
            self.data = self.pivot[self.day:self.day+1]

            self.price = self.data.loc[:, ['adj_close']].values[0].tolist()
            self.ta = self.data.loc[:, self.TA_columns].values[0].tolist()

            self.state = self.value.tolist() + self.price + self.qty + self.ta

            end_total_asset = self.value + sum(np.array(self.price) * np.array(self.qty))
            self.reward = (end_total_asset - begin_total_asset)[0]
            self.asset_memory.append(end_total_asset[0])
            # print("end_total_asset",end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def render(self, mode='human'):
        # print("self.value", self.value,)

        # print("Step: {:05} | Date : {} | Cash: {:8.2f} | Portfolio: {:8.2f} | Reward: {:>4.2f}".format(
        #    self.day, self.dates[self.day], self.value[0], self.asset_memory[-1], self.reward))
        # print(np.shape(self.reward))

        line = [self.modelName, self.day, str(self.dates[self.day]), str(
            self.value[0]), str(self.asset_memory[-1]), self.reward]
        print("Step", self.day, line)

        # print(self.price, self.qty)
        # prices = str(self.price).strip('[]')
        # qty = str(self.qty).strip('[]')
        #print("type", type(self.price), type(self.qty))

        self.ledger.append(line + self.price + self.qty)
        # self.ledger.append(self.qty)
        # self.ledger.append(line)
        # print(self.state)
        # self.ledger.append(self.state)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
