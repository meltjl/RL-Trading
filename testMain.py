import sys
import csv
import os
import csv
import json
import getopt
import mel

if __name__ == "__main__":
    model_name = "ppo2"
    refreshData = 0
    portfolio = 1

    with open('./config.json', 'r') as f:
        config = json.load(f)

    df = mel.get_data(config, portfolio=portfolio, refreshData=refreshData)

    for lr in [1e-3, 1e-3]:
        mel.testHyperparameters(portfolio, config, df, lr)
