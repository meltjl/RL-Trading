title : cap1
title: Create custom gym environments from scratch — A stock market example
paper   : 01A) https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
code :    01A) https://github.com/notadamking/Stock-Trading-Environment

paper   : 01B) https://towardsdatascience.com/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4
code :    01B) https://github.com/notadamking/Stock-Trading-Visualization

also : see 01 RLTrader << same author

#--------------------------------------------------------------------------------
## Routine stuff
#--------------------------------------------------------------------------------
# Create virtual env
ubuntoo@ubuntoo:~/mnt/share/venv$ python3 -m venv mel


# activate environment
ubuntoo@ubuntoo:~/mnt/share/venv$ source mel/bin/activate


# install requirement file
(mel) ubuntoo@ubuntoo:~/mnt/share/venv$ pip install -r /mnt/share/Code/cap1/_requirements.txt


#--------------------------------------------------------------------------------
## extra library required to run project
#--------------------------------------------------------------------------------
# https://stable-baselines.readthedocs.io/en/master/guide/examples.html
pip install stable_baselines

# code extracted from the deprecated matplotlib.finance for plotting purposes
pip install mpl_finance

# https://quantopian.github.io/pyfolio/
pip install pyfolio

(mel) ubuntoo@ubuntoo:~/mnt/share/venv$ pip freeze > /mnt/share/Code/mel_code/_requirements.txt

#--------------------------------------------------------------------------------
# Train & Test
#--------------------------------------------------------------------------------
# make sure python is run from this directory as the file path is set to relative
# if run snippets within atom, press CTRL + , to go to Setting, scroll down and
# make sure default current directory is set to the "Project Directory of Script"

Steps to run:
1. Type in python main.py. The deualt model is PPO2 and using portfolio1
    (mel) ubuntoo@ubuntoo:/mnt/share/Code/mel$ python main.py

2. To select other data, edit to config.json to configure the combination of assets and dates
    {
    	"api" : xxxxx,
    	"portfolios": [{
    			"name": "portfolio1",
    			"asset": ["IBM"],
    			"start_date": "2018-03-20",
    			"end_date": "None",
    			"commission_fee": "1e-5"
    		},
    		{
    			"name": "portfolio2",
    			"asset": ["IBM", "GE", "BA", "MMM", "ABT", "CA"],
    			"start_date": "2010-01-01",
    			"end_date": "None",
    			"commission_fee": "1e-5"
    		}
    	]
    }

3. To run specific portfolio by different algorithm. See below. Note index starts from zero.
    a) To use portfolio2 using PPO2 :
    (mel) ubuntoo@ubuntoo:/mnt/share/Code/mel$ python main.py -m PPO2 -p 1

    b) To use portfolio1 using DDPG :
    (mel) ubuntoo@ubuntoo:/mnt/share/Code/mel$ python main.py -m DDPG -p 0
