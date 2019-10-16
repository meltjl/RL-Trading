

## Libraries
See file _requirements.txt



## Runnign the code
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






