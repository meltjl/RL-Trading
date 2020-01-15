
# Reinforcement Learning in Trading
This project is completed as part of the Master of Data Science degree at the University of Sydney, Australia. 
<ul>
<li>Supervisor : <a href='https://sydney.edu.au/engineering/about/our-people/academic-staff/matloob-khushi.html'>Dr Matloob Khushi, Director, Master of Data Science School of Computer Science</a>.</li>

<li>Unit of Study : <a href='https://cusp.sydney.edu.au/students/view-unit-page/alpha/COMP5707'>COMP5707: Information Technology Capstone A</a></li>

<li>Date of completion : Dec 2019</li>
<ul>

## Overview
The idea of predicting financial instruments has been the goal of many due in part to the
expectation that predicting these instruments can prove lucrative. Whilst the accurate
prediction of price seemed reasonable, they do not necessarily guarantee positive returns due
to commissions, large profit draw-downs and excessive switching behaviours. Reinforcement
Learning (RL) is an autonomous approach to decision making process through repetitive self-
learning and evaluation. The idea is to train an agent to learn to execute an order by acting on
a suitable strategy that maximizes profit.

In this project, we adapted the <a href='https://github.com/hust512/DQN-DDPG_Stock_Trading'>codes</a> from <a href='https://arxiv.org/abs/1811.07522v1'>Practical Deep Reinforcement Learning Approach for Stock Trading, Xiong et al (2018)</a> but applied the <a href='https://arxiv.org/abs/1707.06347'>Proximal Policy Optimization Algorithmm, Schulman et al (2017)</a>. The model achieved an annual return of 34.06%. We also found that adding technical indicators altered the agent’s trading activities significantly.


## Results
<figure>
  <img src="https://github.com/meltjl/RL-Trading/blob/master/result/Figure1.png" alt="Figure1" width="75%" height="75%">
  <figcaption>Fig.1 - Comparison of before and after training data set using PPO2 with different clipping (0% commission and without technical indicators)</figcaption>
</figure>
<hr>
<figure>
  <img src="https://github.com/meltjl/RL-Trading/blob/master/result/Figure2.png" alt="Figure2" width="75%" height="75%">
  <figcaption>Fig.2 - Comparison of portfolio value when technical indicators are used (0% commission)</figcaption>
</figure>
<hr>
<figure>
  <img src="https://github.com/meltjl/RL-Trading/blob/master/result/Figure3.png" alt="Figure3"  width="75%" height="75%">
  <figcaption>Fig.3 - Buy and Sell activity for test data (No technical indicators) shows the lack of buy/sell actvities</figcaption>
</figure>
<br><br><br>
<figure>
  <img src="https://github.com/meltjl/RL-Trading/blob/master/result/Figure4.png" alt="Figure4" width="75%" height="75%">
  <figcaption>Fig.4 - Buy and Sell activity for test data (With technical indicators) shows increased buy/sell activities</figcaption>
</figure>
<br><br><br>
<hr>
<figure>
  <img src="https://github.com/meltjl/RL-Trading/blob/master/result/Figure5.png" alt="Figure5"  width="75%" height="75%">
  <figcaption>Fig.5 - Comparison of Test result using PPO2 algorithm under various commission rates.</figcaption>
</figure>

<hr>

## Environment
### Create virtual environment
<pre>$ python3 -m venv RL_Trading</pre>

### Activate environment
<pre>$ source RL_Trading/bin/activate</pre>

### Install all dependent libraries
<pre>(RL_Trading) $ pip install -r requirements.txt</pre>

### Reproducability
The following files were edited to ensure reproducibility. Copy the files from
<ul>
<li>/files To Overwrite/stable_baselines/common/policies.py</li>
<li>/files To Overwrite/stable_baselines/ppo2/ppo2.py</li>
</ul>

to your virtual environment.
<ul>
<li>../lib/python3.6/site-packages/stable_baselines/common/policies.py</li>
<li>../lib/python3.6/site-packages/stable_baselines/ppo2/ppo2.py</li>
</ul>


### Data
1. At bare minimum, data set must contain at least three columns with the columns name and date format defined exactly below:
    <pre>
    ticker, date, adj_close
    AAPL, 2000-01-03, 111.9375
    AXP, 2000-01-03, 157.25
    </pre>

2. To change commission rate or add other data, edit to config.json to configure the combination of assets and dates
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


### Running the code
Example 1 : To run portfolio4 without technical indicators

<pre>$ python main.py -p 4 -t N</pre>

<pre>*** Run agent on unseen data ***
observation_space :	 Box(57,)
action_space :	 Box(28,)
**** Summary*****
Model:			 PPO2_PORTFOLIO4_20191211 2139_0_TEST LR=0.01, CLIPRANGE=0.3, COMMISSION=0.00
Number of Assets:	      28
Initial Investment :	10000.00
Portfolio Value:	22090.92
% Returns:		  120.91%
***************

train_dates: 2000-01-03 00:00:00 2015-12-31T00:00:00
test_dates: 2016-01-04T00:00:00 2018-09-21T00:00:00
backtest 0 : SUM reward : before | after | backtest :  17276.22 |  91000.59 |  12217.08
</pre>

Example 2 : To use portfolio4 with technical indicators
<pre>$ python main.py -p 4 -t Y</pre>

<pre>*** Run agent on unseen data ***
observation_space :	 Box(561,)
action_space :	 Box(28,)
**** Summary*****
Model:			 PPO2_PORTFOLIO4_20191211 2140_0_TEST LR=0.01, CLIPRANGE=0.3, COMMISSION=0.00
Number of Assets:	      28
Initial Investment :	10000.00
Portfolio Value:	19298.47
% Returns:		   92.98%
***************

train_dates: 2000-01-03 00:00:00 2015-12-31T00:00:00
test_dates: 2016-01-04T00:00:00 2018-09-21T00:00:00
backtest 0 : SUM reward : before | after | backtest :  16377.49 |  14912.09 |  9364.97

</pre>
