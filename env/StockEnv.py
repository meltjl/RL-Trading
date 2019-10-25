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


class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, logfile, modelName, initial_investment=10000, seed=7, commission=0, addTA=False):
        super(StockEnv, self).__init__()
        self.addTA = addTA
        self.dates = df.date.unique()
        self.numSecurity = len(df.ticker.unique())
        self.numTrainDay = len(self.dates)
        self.terminal = False
        self.commission = float(commission)*-1

        self.initial_investment = initial_investment
        self.logfile = logfile
        self.ledger = []
        self.modelName = modelName
        self._seed(seed)

        # create a place holder to store weight/qty
        df.loc[:, 'qty'] = 0

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
        self.reward = 0
        self.commission_paid = [0] * len(self.dates)
        self.transaction = {"buy_amt": 0, "buy_commission": 0,
                            "sell_amt": 0, "sell_commission": 0, }

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
            sell_amt = self.price[index] * quantity

            self.transaction["sell_amt"] += sell_amt
            self.transaction["sell_commission"] += sell_amt * self.commission

            self.commission_paid[self.day] += sell_amt * self.commission
            self.value += sell_amt + (sell_amt*self.commission)
            self.qty[index] -= quantity

            # update investment and qty
            self.state = self.value.tolist() + self.price + self.qty + self.ta
        else:
            # print("No asset to sell")
            pass

    def _buy_stock(self, index, action):
        min_quantity = self.state[0] // self.price[index]
        quantity = min(min_quantity, action)
        buy_amt = self.price[index] * quantity

        self.transaction["buy_amt"] += buy_amt
        self.transaction["buy_commission"] += buy_amt * self.commission

        self.commission_paid[self.day] += buy_amt * self.commission

        self.value -= (buy_amt - (buy_amt * self.commission))
        self.qty[index] += quantity

        # print("buy", (self.price[index] * quantity) * self.commission)
        # print(self.commission_paid[self.day])
        # print(type(self.value.tolist()), type(self.price), type(self.qty), type(self.ta))
        # update investment and qty
        self.state = self.value.tolist() + self.price + self.qty + self.ta

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

            # file = open(self.logfile, 'a+')
            # file.write(','.join(self.ledger))
            # file.write(self.ledger)

            # x = self.ledger + [self.pivot.loc[:, self.TA_columns].values.tolist()]
            # print("zzzzz")
            # print(self.pivot.loc[:, self.TA_columns].head())
            # print(x)
            with open(self.logfile, 'a+') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(self.ledger)

            return self.state, self.reward, self.terminal, {}

        else:
            begin_total_asset = (self.value + sum(np.array(self.price) * np.array(self.qty)))
            begin_cash = self.state[0]
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

            self.portfolio_value.append(end_total_asset[0])
            # self.net_portfolio_value.append(end_total_asset[0]+self.commission_paid[self.day-1])
            # print("end_total_asset",end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def render(self, mode='human'):
        # print("self.value", self.value,)

        # print("Step: {:05} | Date : {} | Cash: {:8.2f} | Portfolio: {:8.2f} | Reward: {:>4.2f}".format(
        #    self.day, self.dates[self.day], self.value[0], self.portfolio_value[-1], self.reward))
        # print(np.shape(self.reward))

        # self.commission_paid is negative
        #net_reward = self.portfolio_value[-1] + self.commission_paid[self.day-1]
        #print("self.transaction", self.transaction)
        line = [self.modelName, self.addTA, self.day, str(self.dates[self.day]), str(
            self.value[0]), str(self.portfolio_value[-1]), self.reward,
            self.commission_paid[self.day-1],
            self.transaction["buy_amt"], self.transaction["buy_commission"],
            self.transaction["sell_amt"], self.transaction["sell_commission"]
        ]
        # print("Step", self.day, line)

        self.ledger.append(line + self.price + self.qty)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
