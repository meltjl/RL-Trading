import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
import csv
from gym import spaces
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import datetime
from calendar import isleap

# player mode - slightly mor advance with more functinality


class StockEnvPlayer(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, logfile, modelName, initial_investment=10000, seed=7, commission=0, addTA=False):
        super(StockEnvPlayer, self).__init__()
        self.addTA = addTA
        self.dates = df.date.unique()
        self.numSecurity = len(df.ticker.unique())
        self.numTrainDay = len(self.dates)
        self.terminal = False
        self.commission = float(commission)*-1

        self.initial_investment = initial_investment
        self.logfile = logfile

        self.modelName = modelName
        self._seed(seed)

        # create a place holder to store weight/qty
        df.loc[:, 'qty'] = 0.0

        # all columns after adjclose are considered TA
        self.TA_columns = df.columns[df.columns.get_loc("adj_close")+1:-1]
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
        with open(logfile, 'a+') as f:
            ap = ['asset' + str(i) + '_price' for i in range(self.numSecurity)]
            aq = ['asset' + str(i) + '_qty' for i in range(self.numSecurity)]
            others = ['asset' + str(i) + '_' + col for i in range(self.numSecurity)
                      for col in self.TA_columns]
            column = 'model, incTA?, step, date, cash, portfolio, reward, total_commission, buy_amt, buy_commission, sell_amt, sell_commission,' + \
                ','.join(ap) + ',' + ','.join(aq) + ',' + '\n'
            f.write(column)

    def reset(self):
        self.portfolio_value = [self.initial_investment]
        self.net_cash = [self.initial_investment]
        self.day = 0
        self.ledger = []
        self.reward = 0
        self.commission_paid = [0] * len(self.dates)
        self.transaction = {"buy_amt": 0, "buy_commission": 0,
                            "sell_amt": 0, "sell_commission": 0, }

        # return as range of one day else it becomes a panda series
        self.data = self.pivot[self.day:self.day+1]
        self.value = self.data.loc[:, ['initial']].values[0]
        self.price = self.data.loc[:, ['adj_close']].values[0]
        self.qty = self.data.loc[:, ['qty']].values[0]
        self.ta = self.data.loc[:, self.TA_columns].values[0]
        self.state = np.concatenate((self.value, self.price, self.qty, self.ta))

        return self.state

    def _sell_stock(self, index, action):
        if self.qty[index] > 0:
            quantity = min(abs(action), self.qty[index])
            #last_price = self.pivot.loc[self.day-2:self.day-1, ['adj_close']].values[0][index]

            # stop loss if price drop more than 5%
            # if (self.price[index] / last_price - 1) < -0.05:
            #print("\n***STOP LOSS***", index, self.dates[self.day], quantity,  self.qty[index])
            #quantity = self.qty[index]

            sell_amt = self.price[index] * quantity
            self.transaction["sell_amt"] += sell_amt
            self.transaction["sell_commission"] += sell_amt * self.commission

            self.commission_paid[self.day] += sell_amt * self.commission
            self.value += sell_amt + (sell_amt*self.commission)
            self.qty[index] -= quantity

            # update investment and qty
            self.state = np.concatenate((self.value, self.price, self.qty, self.ta))
        else:
            # print("No asset to sell")
            pass

    def _buy_stock(self, index, action):
        # keep at least 10 of cash

        min_quantity = self.state[0] // self.price[index]

        quantity = min(min_quantity, action)
        buy_amt = self.price[index] * quantity
        # print("buy", self.dates[self.day], index, action, quantity)

        self.transaction["buy_amt"] += buy_amt
        self.transaction["buy_commission"] += buy_amt * self.commission

        self.commission_paid[self.day] += buy_amt * self.commission

        self.value -= (buy_amt - (buy_amt * self.commission))
        self.qty[index] += quantity

        # update investment and qty
        self.state = np.concatenate((self.value, self.price, self.qty, self.ta))

    def step(self, actions):
        self.terminal = self.day >= (self.numTrainDay-1)
        self.transaction = {"buy_amt": 0, "buy_commission": 0,
                            "sell_amt": 0, "sell_commission": 0, }
        if self.terminal:

            print("**** Summary*****")
            print("Model:\t\t\t", self.modelName.upper())
            print("Number of Assets:\t{:8.0f}".format(self.numSecurity))
            print("Initial Investment :\t{:8.2f}".format(self.initial_investment))

            portfolio_value = self.state[0] + sum(np.array(self.price) * np.array(self.qty))
            rtns_dollar = round(portfolio_value - self.initial_investment, 2)
            rtns_pct = round((portfolio_value/self.initial_investment-1)*100, 2)
            # rtns_annualised = (1+rtns_pct) ** (1/self.years)-1

            print("Portfolio Value:\t{:8.2f}".format(portfolio_value))
            print("% Returns:\t\t{:8.2f}%".format(rtns_pct))
            print("***************")

            fig, ax = plt.subplots()
            ax.set_title(self.modelName)
            ax.set_ylabel('Total Asset $')
            ax.set_xlabel('Episode')
            ax.plot(self.portfolio_value, color='tomato')
            plt.savefig('image/{}.png'.format(self.modelName))
            plt.close()

            with open(self.logfile, 'a+') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(self.ledger)

            return self.state, self.reward, self.terminal, {}

        else:
            begin_total_asset = self.value + np.sum(self.price * self.qty)
            begin_cash = self.state[0]

            # actions are predicted by the RL algo to spit out the quantity to buy/sell
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[:: -1][: np.where(actions > 0)[0].shape[0]]
            #print("sell|buy index", actions, sell_index, buy_index)

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            # get next day's price & ta
            self.day += 1
            self.data = self.pivot[self.day:self.day+1]

            self.price = self.data.loc[:, ['adj_close']].values[0]
            self.ta = self.data.loc[:, self.TA_columns].values[0]
            self.state = np.concatenate((self.value, self.price, self.qty, self.ta))
            end_total_asset = self.value + np.sum(self.price * self.qty)

            self.reward = (end_total_asset - begin_total_asset)[0]
            self.portfolio_value.append(end_total_asset[0])

        return self.state, self.reward, self.terminal, {}

    def render(self, mode='human'):
        line = np.array([self.modelName, self.addTA, self.day, str(self.dates[self.day]), str(
            self.value[0]), str(self.portfolio_value[-1]), self.reward,
            self.commission_paid[self.day-1],
            self.transaction["buy_amt"], self.transaction["buy_commission"],
            self.transaction["sell_amt"], self.transaction["sell_commission"]
        ])
        display = np.concatenate((line, self.price, self.qty))
        self.ledger.append(display)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
