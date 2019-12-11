

# Overview
The idea of predicting financial instruments has been the goal of many due in part to the
expectation that predicting these instruments can prove lucrative. Whilst the accurate
prediction of price seemed reasonable, they do not necessarily guarantee positive returns due
to commissions, large profit draw-downs and excessive switching behaviours. Reinforcement
Learning (RL) is an autonomous approach to decision making process through repetitive self-
learning and evaluation. The idea is to train an agent to learn to execute an order by acting on
a suitable strategy that maximizes profit.

In this project, we adapted the codes from <a href='https://www.groundai.com/project/practical-deep-reinforcement-learning-approach-for-stock-trading/'>Xiong et al (2018)</a> but applied the <a href:'https://arxiv.org/abs/1707.06347'>Proximal Policy Optimization Algorithmm Schulman et al (2017)</a>. The model achieved an annual return of 34.06%. We also found that adding technical indicators altered the agent’s trading activities significantly.

# Results




# Environment
## Create virtual environment
<pre>$ python3 -m venv RL_Trading</pre>

## Activate environment
<pre>$ source RL_Trading/bin/activate</pre>

## Install all dependencies libraries
<pre>(RL_Trading) $ pip install -r requirements.txt</pre>

## Reproducability
The following files were edited to ensure reproducibility. Copy the files from
<ul>
<li>/files To Overwrite/stable_baselines/common/policies.py</li>
<li>/files To Overwrite/stable_baselines/ppo2/ppo2.py</li>
<ul>

to your virtual environment.
<ul>
<li>../lib/python3.6/site-packages/stable_baselines/common/policies.py</li>
<li>../lib/python3.6/site-packages/stable_baselines/ppo2/ppo2.py</li>
<ul>


## Data
At bare minimum, data set must contain at least three columns with the columns name and date format defined exactly below:
ticker	date	    adj_close
AAPL	2000-01-03	111.9375
AXP	    2000-01-03	157.25


## Running the code
Steps to run:
1. To run specific portfolio add the -p parameter
    a) To use portfolio4 without technical indicators
    <pre>$ python main.py -p 4 -t N<</pre>

    b) To use portfolio3 with technical indicator
    <pre>$ python main.py -p 3 -t Y</pre>


2. To select other data, edit to config.json to configure the combination of assets and dates
    <pre>
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
    </pre>
