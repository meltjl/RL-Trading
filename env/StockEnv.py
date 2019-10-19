import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

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
        self.train_daily_data = []
        self.initial_investment = initial_investment
        self.logfile = logfile
        self.ledger = []
        self.modelName = modelName

        for date in np.unique(df.date):
            self.train_daily_data.append(df[df.date == date])

        self.day = 0

        # buy or sell maximum 5 shares
        self.action_space = spaces.Box(
            low=-self.numSecurity, high=self.numSecurity, shape=(self.numSecurity,), dtype=np.int8)

        # [money]+[prices for each asset]+[owned shares for each asset]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1 + self.numSecurity*2,))

        print('observation_space :\t', self.observation_space)
        print('action_space :\t', self.action_space)

        self._seed(seed)

    def reset(self):
        self.asset_memory = [self.initial_investment]
        self.day = 0
        self.reward = 0
        self.data = self.train_daily_data[self.day]
        # [money]+[prices for each asset]+[owned shares for each asset]
        self.state = [self.initial_investment] + \
            self.data.adj_close.values.tolist() + [0 for i in range(self.numSecurity)]
        # print(self.state)
        return self.state

    def _sell_stock(self, index, action):
        if self.state[index+(self.numSecurity+1)] > 0:
            quantity = min(abs(action), self.state[index+(self.numSecurity+1)])
            self.state[0] += self.state[index+1] * \
                min(abs(action),  self.state[index+(self.numSecurity+1)])
            self.state[index+(self.numSecurity+1)] -= min(abs(action),
                                                          self.state[index+(self.numSecurity+1)])

            # print("S {:2.5f} unit of asset {} index @ {:4.2f}".format(quantity, index, self.state[index+1]))

        else:
            # print("No asset to sell")
            pass

    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index+1]
        quantity = min(available_amount, action)
        amount_spend = quantity * self.state[index+1]

        self.state[0] -= self.state[index+1] * min(available_amount, action)

        # print("available_amount", available_amount)
        # print("action", action)

        self.state[index+(self.numSecurity+1)] += quantity
        # print("B {:2.5f} unit of asset {} @ {:4.2f} = {:4.2f}".format(quantity, index, self.state[index+1], amount_spend))

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

            portfolio_value = self.state[0] + sum(np.array(self.state[1:(
                self.numSecurity+1)])*np.array(self.state[(self.numSecurity+1):]))  # - self.initial_investment
            print("Portfolio Value:\t{:8.2f}".format(portfolio_value))
            print("% Returns:\t\t{:8.2f}%".format((portfolio_value/self.initial_investment-1)*100))
            print("***************")

            file = open(self.logfile, 'a+')
            file.write(','.join(self.ledger))

            return self.state, self.reward, self.terminal, {}

        else:
            asset_prices = np.array(self.state[1:(self.numSecurity+1)])
            asset_quantity = np.array(self.state[(self.numSecurity+1):])
            begin_total_asset = self.state[0] + sum(asset_prices * asset_quantity)
            # print("begin_total_asset:{:.2f}".format(begin_total_asset))

            # actions are predicted by the RL algo to spit out the quantity to buy/sell
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.train_daily_data[self.day]
            self.state = [self.state[0]] + self.data.adj_close.values.tolist() + \
                list(self.state[(self.numSecurity+1):])

            # get next days prices
            asset_prices = np.array(self.state[1:(self.numSecurity+1)])
            asset_quantity = np.array(self.state[(self.numSecurity+1):])
            #print("asset_prices", asset_prices)
            #print("asset_quantity", asset_quantity)

            end_total_asset = self.state[0] + sum(asset_prices * asset_quantity)
            self.reward = end_total_asset - begin_total_asset
            self.asset_memory.append(end_total_asset)
            # print("end_total_asset",end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def render(self, mode='human'):
        print("Step: {:05} | Date : {} | Cash: {:8.2f} | Portfolio: {:8.2f} | Reward: {:>4.2f}".format(
            self.day, self.dates[self.day], self.state[0], self.asset_memory[-1], self.reward))

        line = '\n{}, {},{},{},{},{},'.format(self.modelName, self.day, str(
            self.dates[self.day]), self.state[0], self.asset_memory[-1], self.reward)

        prices = str(self.state[1:(self.numSecurity+1)]).strip('[]')
        qty = str(self.state[(self.numSecurity+1):]).strip('[]')
        self.ledger.append(line + prices + ',' + qty)

        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
