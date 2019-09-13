import glob
import json
import sys
import itertools
import collections
import time, datetime
import pandas as pd
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm


from pylab import rcParams
rcParams['figure.figsize'] = 20, 4


pd.set_option('precision', 12)


MARKETS = ["ETHBTC", "BTCUSDT", "ETHUSDT", "BNBBTC", "BNBETH", "BNBUSDT"]
FOLDER = 'data/L2'
print('FOLDER', FOLDER)


def snapshots(pair):
    return sorted(glob.glob(f'{FOLDER}/{pair}/2019*snapshot.json'))


def updates(pair):
    BTs = sorted(glob.glob(f'{FOLDER}/{pair}/2019*update.json'))
    return BTs


def trades_file(pair):
    BTs = sorted(glob.glob(f'{FOLDER}/{pair}/2019*trades.json'))
    return BTs

def sample(of_file):
    return json.load(open(of_file))[0]

def tradesframe(file):
    ts = pd.DataFrame(json.load(open(file)))
    if ts.empty:
        return ts
    ts = ts.drop(['M', 's', 'e', 'a'], axis=1).astype(np.float64)
    ts['t'] = 1 + ts['E'] // 1000
    ts['delay'] = ts['E'] - ts['T']
    ts['num'] = ts['l'] - ts['f'] + 1
    ts['up'] = 1 - 2*ts['m']
    ts = ts.drop(['E', 'T', 'f', 'l', 'm'], axis=1).set_index(['t'])
    return ts


def json2df(js, mult=1000):
    try:
        df = pd.DataFrame(js, columns=['price', 'size']).astype(np.float64).set_index('price')
    except:
        df = pd.DataFrame(js, columns=['price', 'size', 'none']).drop(['none'], axis=1).astype(np.float64).set_index('price')
    #df.index = (df.index * mult).astype(int)
#    assert((df.index > 4).all())
    return df

def merge(df, upd_df, is_ask):
    df = df.append(upd_df)
    df = df[~df.index.duplicated(keep='last')]
    df = df[(df!=0).any(axis=1)]
#    assert((df.index > 4).all())
    return df.sort_index(ascending=is_ask)

def price(bids, asks):
    bbp, bbs = bids.index[0], bids['size'].iloc[0]
    bap, bas = asks.index[0], asks['size'].iloc[0]
    price = (bbp * bas + bap * bbs) / (bbs + bas)
    return round(price, 8)


def buildL2(snapshot_file):
    snap = json.load(open(snapshot_file))
    lastUpdateId = snap['lastUpdateId']
    #print(snapshot_file, "lastUpdateId:", lastUpdateId)
    bids = json2df(snap['bids'])
    asks = json2df(snap['asks'])
    return bids, asks, lastUpdateId



def replayL2(pair, emaNew=1/256):
    snapshotupdates = {}
    for snapshot_file in snapshots(pair):
        snap = json.load(open(snapshot_file))
        lastUpdateId = snap['lastUpdateId']
        snapshotupdates[lastUpdateId] = snapshot_file
    snapshotupdates = iter(sorted(snapshotupdates.items()))
    next_snap,snapshot_file = next(snapshotupdates)

    bids, asks, lastUpdateId = buildL2(snapshot_file)

#    textsnap = Label("Hello")
#    textupds = Label("World")
    
    emaPrice = None
    file_updates = tqdm(updates(pair))
    for fupdate in file_updates:
#        cache = fupdate.replace('update.json', 'cache.h5')
#        with pd.HDFStore(cache) as store:
#            try:
#                #for (path, subgroups, subkeys) in tqdm(store.walk(), leave=False):
#                #    if path:
#                #        yield store.get(path+'/b'), store.get(path+'/a'), store.get(path+'/p'), store.get(path+'/t')
#                pass
#            except:
#                pass
            ftrades = fupdate.replace('update', 'trades')
            file_updates.set_description(fupdate + " " + ftrades)
#            textsnap.value = fupdate+" & "+ftrades
            alltrdf = tradesframe(ftrades)
            js = json.load(open(fupdate))
            allupdates = tqdm(js, leave=False)
            prev_ts = None
            for upds in allupdates:
                if upds['e'] != 'depthUpdate':
                    continue
                U = upds['U']
                u = upds['u']
                E = upds['E']
                ts = 1 + E // 1000

                if u >= next_snap:
                    bids, asks, lastUpdateId = buildL2(snapshot_file)
                    next_snap,snapshot_file = next(snapshotupdates)

                ub = json2df(upds['b'])
                ua = json2df(upds['a'])
                #print(asks.head())
                bids = merge(bids, ub, False)
                asks = merge(asks, ua, True)
                bid, ask = bids.index[0], asks.index[0]
                if bid > ask:
                    print(u, bid, ">", ask, fupdate, datetime.datetime.fromtimestamp(ts))
                    bids, asks = bids[bids.index < ask], asks[asks.index > bid]
                bid, ask = bids.index[0], asks.index[0]
                px = price(bids, asks)
                if not alltrdf.empty:
                    trdf = alltrdf.iloc[alltrdf.index==ts]
                    trdf = trdf.set_index(['p'])
                    trdf.loc[px] = 0
                    trdf.sort_index(inplace=True)
                    prev_px = px
                else:
                    trdf = trdf.loc[[prev_px]]
                emaPrice = px * emaNew + (emaPrice if emaPrice is not None else px) * (1-emaNew)
#                textupds.value = f"{datetime.datetime.fromtimestamp(ts)}  = {ts},   u={u}<{next_snap}  px={px}"
                assert ask >= bid, f"{bid} > {ask}\n{ub}\n{ua}\n{bids}\n{asks}"
                oneSec = bids.cumsum(), \
                        asks.cumsum(), \
                        pd.Series({'time': ts, 'price': px, 'emaPrice': emaPrice, 'bid': bid, 'ask': ask}), \
                        trdf

                allupdates.set_description(f"ts={ts}, E={E}, trades={len(trdf)}")
                # assert not prev_ts or ts == 1 + prev_ts
                prev_ts = ts

                yield oneSec





FRAC_LEVELS = 0.01
NUM_LEVEL_BINS = 128
SPACING = np.cumsum(0+np.linspace(0, NUM_LEVEL_BINS, NUM_LEVEL_BINS, endpoint=False))# * 4
SPACING = SPACING / SPACING[-1]
#print(SPACING)
def bin_books(dfb, dfa, tr, ref_price, zoom_frac=FRAC_LEVELS, spacing=SPACING):
    b_idx = np.round(pd.Index(ref_price * (1-spacing*zoom_frac)), 7)
    a_idx = np.round(pd.Index(ref_price * (1+spacing*zoom_frac)), 7)

    t_idx = b_idx[::-1].append(a_idx)
    
    reind_b = dfb.reindex(t_idx[::-1], method='ffill', fill_value=0)[::-1]
    reind_a = dfa.reindex(t_idx, method='ffill', fill_value=0)
    treind_b = tr[tr['up']<=0].groupby(level=0).sum()[::-1].cumsum()[::-1].reindex(t_idx, method='bfill', fill_value=0).diff(-1).fillna(0)
    treind_a = tr[tr['up']>=0].groupby(level=0).sum().cumsum().reindex(t_idx, method='ffill', fill_value=0).diff().fillna(0)
    
    treind_b = np.arcsinh(treind_b)
    treind_a = np.arcsinh(treind_a)
    reind_b = np.arcsinh(reind_b)
    reind_a = np.arcsinh(reind_a)
    return reind_b, reind_a, treind_b, treind_a



def sampleArrays(market):
    replayer = replayL2(market, emaNew=1/64)
    arrs = []
    trrs = []
    pric = []
    prev_price = None
    i = 0
    spacing = np.arange(64)
    #spacing = np.square(spacing) + spacing
    spacing = spacing / spacing[-1]
    for bi,ai,tpi,tri in replayer:
        prev_price = prev_price or tpi['price']
        bib,aib,trb,tra  = bin_books(bi,ai,tri, ref_price=prev_price, zoom_frac=1/256, spacing=spacing)
        prev_price = tpi['emaPrice']#0.5*(tpi['bid'] + tpi['ask'])
        arrs.append(np.concatenate([bib.values - aib.values]))
        trrs.append(np.concatenate([trb.values - tra.values]))
        pric.append(np.array([tpi['price'], tpi['emaPrice'], tpi['bid'], tpi['ask'], tpi['time']]))
        i += 1
        if i > 512:
            break
    books = np.stack(arrs)
    prices = np.stack(pric)
    trades = np.stack(trrs)
    return {'books': books, 'prices': prices, 'trades':trades}

def sampleImages(books, prices, trades):
    print(books.shape, prices.shape, trades.shape)
    plt.margins(0.0)
    plt.plot(prices[:, 0])
    plt.plot(prices[:, 1])
    plt.plot(prices[:, 2])
    plt.plot(prices[:, 3])
    plt.show()
    nth = 0
    im = np.abs(books[:, :, 0]).copy()
    im[im == 0] = -1
    plt.imshow(im.T, cmap='nipy_spectral', origin="lower")
    plt.show()
    im = np.abs(trades[:, :, 2])
    #im[im == 0] = 3
    plt.imshow(im.T, cmap='nipy_spectral', origin="lower")
    plt.show()
    im0 = books[:, :, 0].T/10#20
    im1 = trades[:, :, 0].T/1
    im2 = trades[:, :, 2].T/1
    im3 = np.stack([im0, im1, im2], -1)+0.5
    print(im3.shape)
    plt.imshow(im3[:,:,:], origin="lower")
    plt.show()
    

def build_time_level_trade(books, prices, filename='data/timeUpDn.npy'):
    pricestep = 0.000001
    sidesteps = books.shape[1] // 2
    FUTURE = 1200*10
    timeUpDn = np.zeros_like(books[:, :2*sidesteps, :1]) + FUTURE
    #########################################################
                            #######
    for i in tqdm(range(prices.shape[0])):
                            #######
        #########################################################
        timeupdn = []
        for j in range(sidesteps):
            thresh = j * pricestep
            p, e, b, a, t = prices[i]
            waitUp = prices[i:i+FUTURE, 2] < a + thresh
            waitDn = prices[i:i+FUTURE, 3] > b - thresh
            timeUp = np.argmin(waitUp) or FUTURE*10
            timeDn = np.argmin(waitDn) or FUTURE*10
            timeupdn.insert(0, [timeDn])
            timeupdn.append([timeUp])
        timeUpDn[i] = timeupdn
    np.save(filename, timeUpDn.astype(np.float32))


def multireplayL2(pairs, emaNew=1/64):
    gens = {pair: replayL2(pair, emaNew) for pair in pairs}
    nexs = {pair: next(gens[pair]) for pair in pairs}
    def secs(pair): return nexs[pair][2]['time']
    tall = max([secs(p) for p in pairs])
    curs = {pair: nexs[pair] for pair in pairs}
    print('tall', tall)
    while True:
        for pair in pairs:
            while secs(pair) < tall:
                curs[pair] = nexs[pair]
                try:
                    nexs[pair] = next(gens[pair])
                except StopIteration:
                    return#break
        yield curs
        tall += 1



def gen_array(markets, width_per_side=64, zoom_frac=1/256):
    market_replay = multireplayL2(markets)
    prev_price = {p: None for p in markets}
    spacing = np.arange(width_per_side)
    #spacing = np.square(spacing) + spacing
    spacing = spacing / spacing[-1]
    spacing = np.arcsin(spacing)*3-spacing*2
    for second in market_replay:
        market_second = collections.defaultdict(list)
        for pair in markets:
            bi,ai,tpi,tri = second[pair]
            prev_price[pair] = prev_price[pair] or tpi['price']
            bib,aib,trb,tra  = bin_books(bi,ai,tri, ref_price=prev_price[pair], zoom_frac=zoom_frac, spacing=spacing)
            prev_price[pair] = tpi['emaPrice']
            arr0 = bib.values - aib.values
            arr1 = tra.values - trb.values
            tp = np.array([tpi['price'], tpi['emaPrice'], tpi['bid'], tpi['ask'], tpi['time']])
            arr3d = np.concatenate([arr0, arr1[:,::2]], axis=-1)
            market_second[pair] = {'ps': [tp], 'bs': [arr3d]}
        yield market_second


def build(total, element):
    for market,second in element.items():
#        print('total', total[market]['ps'][-1])
#        print('element', element[market]['ps'][-1])
        datetotal = datetime.datetime.fromtimestamp(int(total[market]['ps'][-1][-1])).date()
        dateeleme = datetime.datetime.fromtimestamp(int(element[market]['ps'][-1][-1])).date()
        newDay = datetotal < dateeleme
        for name,arrs in second.items():
            if newDay:
                arrday = np.stack(total[market][name]).astype(np.float32)
                np.save(f'data/{datetotal}-{market}-{name}.npy', arrday)
                total[market][name] = element[market][name]
            else:
                total[market][name] += element[market][name]
    return total


def accumulate_array(markets):
    genacc = itertools.accumulate(gen_array(markets), build)
    return genacc
