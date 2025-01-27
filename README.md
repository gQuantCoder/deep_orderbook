# deep_orderbook
## Orderbooks contain so much more organic informations than moving averages... ##

I have seen to many people thinking that machine learning could be applied to finance by feeding a time series into a deep neural network and expecting the contraption to give the next few values.
I can think of a few reasons why this is a very naive approach that would not only be doomed to fail, but also that wouldn't possibly represent the actual data necessary to make decision in automated trading.

This project aims at not only preparing the data in a form that is very adapted to modern image recognition neural network (where recent progress has been spectacular), but also to prepare the gound for a prediction that would be meaningful to the actual dynamic of trading.


(Work in progress on my free time)

## installation

edit ` credentials/coinbase.txt `
```
api_key="organizations/xxxxxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/apiKeys/xxxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx"
api_secret="-----BEGIN EC PRIVATE KEY-----\xxxxxxxxxxxxxxxxx...xxxxxxxxxxxxxxxxxxx\n-----END EC PRIVATE KEY-----\n"
```

  ``` pip install -r requirements.txt ```
  ``` pip install deep_orderbook -e ```

## record data:
```
python deep_orderbook/consumers/recorder.py
```

## visualization
Open a jupyterlab notebook and execute: `live.ipynb` or `replay.ipynb`
machine learning example: `learn.ipynb`


## example of output

![books](https://raw.githubusercontent.com/gQuantCoder/deep_orderbook/master/images/01.png?raw=true "Orderbooks and alpha")
