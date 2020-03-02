FROM python:3.7

COPY . deep_orderbook

RUN pip install --no-cache-dir -r deep_orderbook/requirements.txt

RUN pip install -e deep_orderbook 

CMD [ "python", "./deep_orderbook/deep_orderbook/recorder.py" ]
