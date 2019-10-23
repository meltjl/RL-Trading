

# Environment
## Create virtual environment
<pre>$ python3 -m venv RL_Trading</pre>

## Activate environment
<pre>$ source RL_Trading/bin/activate</pre>

## Install requirement file
<pre>(RL_Trading) $ pip install -r requirements.txt</pre>


## Data
At bare minimum, data set must contain at least three columns with the columns name and date format defined exactly below:
ticker	date	    adj_close
AAPL	2000-01-03	111.9375
AXP	    2000-01-03	157.25


## Running the code
Steps to run:
1. Type in python main.py. The default algorithm is PPO2
    <pre>$ python main.py -p 1</pre>

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

3. To run specific portfolio add the -p parameter. Note index starts from zero.
    a) To use portfolio2:
    <pre>$ python main.py -p 2</pre>

    b) To use portfolio3
    <pre>$ python main.py -p 3</pre>
