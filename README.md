# deep_orderbook
## Orderbooks contain so much more organic informations than moving averages... ##

I've noticed that many people assume machine learning can be applied to finance by simply feeding a time series into a deep neural network, expecting it to predict the next few values. This approach is not only overly simplistic but also fundamentally flawed, as it fails to capture the complexity of financial data and the nuances required for effective decision-making in automated trading.

This project aims to go beyond these limitations by transforming financial data into a format optimized for modern image recognition neural networks—an area that has seen remarkable advancements. Additionally, it lays the foundation for predictions that are aligned with the actual dynamics of trading, making them more relevant and actionable.


(Work in progress on my free time)

## example of output

![books](https://raw.githubusercontent.com/gQuantCoder/deep_orderbook/master/images/01.png?raw=true "Orderbooks and alpha")

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


