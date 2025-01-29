# deep_orderbook
Orderbooks contain so much more organic informations than moving averages... 


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


