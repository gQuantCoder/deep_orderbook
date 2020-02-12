import json
import itertools
import collections
import time, datetime
import pandas as pd
import numpy as np
import asyncio
import aiofiles

from tqdm.auto import tqdm
from deep_orderbook.recorder import MessageDepthCacheManager
import aioitertools

pd.set_option('precision', 12)
import matplotlib.pyplot as plt


class BookShapper:
    PriceShape = [2,3]

    @classmethod
    async def create(cls):
        self = cls()
        self._depth_manager = await MessageDepthCacheManager.create(client=None, loop=None, symbol=None, refresh_interval=None)
        self.sec_trades = dict()
        self.bids = None
        self.asks = None
        self.tpr = None
        self.trdf = None
        self.ts = None
        self.px = None
        self.prev_px = None
        self.emaPrice = None
        self.emaNew = 1/32
        return self

    def on_snaphsot(self, snapshot):
        self.bids, self.asks, lastUpdateId = self.buildL2(snapshot)

    async def on_snaphsot_async(self, snapshot):
        await self._depth_manager._init_cache(snapshot)

    def on_depth_msg(self, msg):
        E = msg['E']
        self.ts = 1 + E // 1000

        ub = self.json2df(msg['b'])
        ua = self.json2df(msg['a'])
        self.bids = self.merge(self.bids, ub, False)
        self.asks = self.merge(self.asks, ua, True)

        bid, ask = self.bids.index[0], self.asks.index[0]
        #assert bid < ask
        if bid > ask:
            print(u, bid, ">", ask, datetime.datetime.fromtimestamp(self.ts))
            self.bids, self.asks = self.bids[self.bids.index < ask], self.asks[self.asks.index > bid]
            bid, ask = self.bids.index[0], self.asks.index[0]
        self.px = self.price(self.bids, self.asks)
        self.emaPrice = self.px * self.emaNew + (self.emaPrice if self.emaPrice is not None else self.px) * (1-self.emaNew)

        self.px += 1e-12
        return self.px

    async def on_depth_msg_async(self, msg):
        await self._depth_manager._depth_event(msg)
        bids, asks = self._depth_manager.get_depth_cache().get_bids_asks()
        ts = self.secondAvail(msg)
        await self.update_ema(bids, asks, ts)

    async def update_ema(self, bids, asks, ts):
        self.ts = ts
        bbp, bbs = bids[0]
        bap, bas = asks[0]
        price = (bbp * bas + bap * bbs) / (bbs + bas)
        self.px = round(price, 8)
        self.emaPrice = self.px * self.emaNew + (self.emaPrice if self.emaPrice is not None else self.px) * (1-self.emaNew)
        self.px += 1e-12
        return self.px

    @staticmethod
    def secondAvail(tr_dict):
        return 1 + tr_dict['E'] // 1000

    async def on_trades_bunch(self, list_trades, force_t_avail=None):
        if force_t_avail:
            self.sec_trades[force_t_avail] = list(list_trades)
            return
        grp_sec = itertools.groupby(list_trades, self.secondAvail)
        for i,l in grp_sec:
            # print(i, list(l))
            self.sec_trades[i] = list(l)
        return

    def on_trades(self, trdf):
        oneSec = self.bids, \
                self.asks, \
                pd.Series({'time': self.ts, 'price': self.px, 'emaPrice': self.emaPrice, 'bid': self.bids.index[0], 'ask': self.asks.index[0]}), \
                trdf
        return oneSec

    async def make_frames_async(self, t_avail, bids=None, asks=None):
        list_trades = self.sec_trades.pop(t_avail, [])
        if list_trades:
            # print("past trades available: ", len(self.sec_trades))
            ts = pd.DataFrame(list_trades).drop(['M', 's', 'e', 'a'], axis=1)
            ts = ts.astype(np.float64)
            #ts['t'] = 1 + ts['E'] // 1000
            ts['delay'] = ts['E'] - ts['T']
            ts['num'] = ts['l'] - ts['f'] + 1
            ts['up'] = 1 - 2*ts['m']
            ts.drop(['E', 'T', 'f', 'l', 'm'], axis=1, inplace=True)
            #ts.set_index(['t'], inplace=True)

            self.trdf = ts.set_index(['p'])
            self.trdf.loc[self.px] = 0
            self.trdf.sort_index(inplace=True)
            self.prev_px = self.px

        else:
            self.trdf = pd.DataFrame(columns=['p', 'q', 'delay', 'num', 'up']).set_index(['p'])
#            self.trdf = self.trdf.loc[[self.prev_px]]

        if bids is None or asks is None:
            bids, asks = self._depth_manager.get_depth_cache().get_bids_asks()
        oneSec = {'time': self.ts,
                  'price': self.px,
                  'bids': pd.DataFrame(bids, columns=['price', 'size']).set_index('price'),
                  'asks': pd.DataFrame(asks, columns=['price', 'size']).set_index('price'),
                  'trades': self.trdf.copy(),
                 'emaPrice': self.emaPrice,
#                 'bid': bids[0][0],
#                 'ask': asks[0][0]
                 }
        return oneSec

    @staticmethod
    def json2df(js):
        try:
            df = pd.DataFrame(js, columns=['price', 'size']).astype(np.float64).set_index('price')
        except:
            df = pd.DataFrame(js, columns=['price', 'size', 'none']).drop(['none'], axis=1).astype(np.float64).set_index('price')
        # assert((df.index > 4).all())
        return df

    @staticmethod
    def merge(df, upd_df, is_ask):
        df = df.append(upd_df)
        df = df[~df.index.duplicated(keep='last')]
        df = df[(df!=0).any(axis=1)]
        # assert((df.index > 4).all())
        return df.sort_index(ascending=is_ask)

    @staticmethod
    def price(bids, asks):
        bbp, bbs = bids.index[0], bids['size'].iloc[0]
        bap, bas = asks.index[0], asks['size'].iloc[0]
        price = (bbp * bas + bap * bbs) / (bbs + bas)
        return round(price, 8)


    def buildL2(self, snapshot):
        lastUpdateId = snapshot['lastUpdateId']
        # print(snapshot_file, "lastUpdateId:", lastUpdateId)
        bids = self.json2df(snapshot['bids'])
        asks = self.json2df(snapshot['asks'])
        return bids, asks, lastUpdateId








    FRAC_LEVELS = 0.01
    NUM_LEVEL_BINS = 128
    SPACING = np.cumsum(0+np.linspace(0, NUM_LEVEL_BINS, NUM_LEVEL_BINS, endpoint=False))# * 4
    SPACING = SPACING / SPACING[-1]
    @staticmethod
    def bin_books(dfb, dfa, tr, ref_price, zoom_frac=FRAC_LEVELS, spacing=SPACING):
        b_idx = np.round(pd.Index(ref_price * (1-spacing*zoom_frac)), 7)
        a_idx = np.round(pd.Index(ref_price * (1+spacing*zoom_frac)), 7)

        t_idx = b_idx[::-1].append(a_idx)
        t_idx_inv = t_idx[::-1]
        
        reind_b = dfb.cumsum().reindex(t_idx_inv, method='ffill', fill_value=0).diff().fillna(0)[::-1]
        reind_a = dfa.cumsum().reindex(t_idx, method='ffill', fill_value=0).diff().fillna(0)
        treind_b = tr[tr['up']<=0].groupby(level=0).sum()[::-1].cumsum().reindex(t_idx_inv, method='ffill', fill_value=0).diff().fillna(0)[::-1]
        treind_a = tr[tr['up']>=0].groupby(level=0).sum().cumsum().reindex(t_idx, method='ffill', fill_value=0).diff().fillna(0)
        
        treind_b = np.arcsinh(treind_b).astype(np.float32)
        treind_a = np.arcsinh(treind_a).astype(np.float32)
        reind_b = np.arcsinh(reind_b).astype(np.float32)
        reind_a = np.arcsinh(reind_a).astype(np.float32)
        return reind_b, reind_a, treind_b, treind_a



    def sampleArrays(self, replayer, numpoints=None, apply_fnct=None):
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
            bib,aib,trb,tra = self.bin_books(bi,ai,tri, ref_price=prev_price, zoom_frac=1/256, spacing=spacing)
            prev_price = tpi['emaPrice']#0.5*(tpi['bid'] + tpi['ask'])
            arrs.append(np.concatenate([bib.values - aib.values]))
            trrs.append(np.concatenate([trb.values - tra.values]))
            pric.append(np.array([tpi['price'], tpi['emaPrice'], tpi['bid'], tpi['ask'], tpi['time']]))
            i += 1
            if apply_fnct and i%10 == 0:
                apply_fnct(**{'books': np.stack(arrs), 'prices': np.stack(pric), 'trades':np.stack(trrs)})
            if numpoints and i >= numpoints:
                break
        books = np.stack(arrs)
        prices = np.stack(pric)
        trades = np.stack(trrs)
        ret = {'books': books, 'prices': prices, 'trades':trades}
        return ret

    def sampleImages(self, books, prices, trades):
        #print(books.shape, prices.shape, trades.shape)
        plt.margins(0.0)
        plt.plot(prices[:, 0])
        plt.plot(prices[:, 1])
        plt.plot(prices[:, 2])
        plt.plot(prices[:, 3])
        plt.show()
        im = np.abs(books[:, :, 0]).copy()
        im[im == 0] = -1
        plt.imshow(im.T, cmap='nipy_spectral', origin="lower")
        plt.show()
        im = np.abs(trades[:, :, 2])
        #im[im == 0] = 3
        plt.imshow(im.T, cmap='nipy_spectral', origin="lower")
        plt.show()
        im0 = books[:, :, 0].T/10
        im1 = trades[:, :, 0].T/1
        im2 = trades[:, :, 2].T/1
        im3 = np.stack([im0, im1, im2], -1)+0.5
        print(im3.shape)
        plt.imshow(im3[:,:,:], origin="lower")
        plt.show()


    @staticmethod
    def gen_array(market_replay, markets, width_per_side=64, zoom_frac=1/256):
        #market_replay = self.multireplayL2(markets)
        prev_price = {p: None for p in range(len(markets))}
        spacing = np.arange(width_per_side)
        #spacing = np.square(spacing) + spacing
        spacing = spacing / spacing[-1]
        spacing = np.arcsin(spacing)*3-spacing*2
        for second in market_replay:
            market_second = {}#collections.defaultdict(list)
            for pair in range(len(markets)):
                bi,ai,tpi,tri = second[pair]
                prev_price[pair] = prev_price[pair] or tpi['price']
                bib,aib,trb,tra  = BookShapper.bin_books(bi,ai,tri, ref_price=prev_price[pair], zoom_frac=zoom_frac, spacing=spacing)
                prev_price[pair] = tpi['emaPrice']
                arr0 = bib.values - aib.values
                arr1 = tra.values - trb.values
                tp = np.array([tpi['price'], tpi['emaPrice'], tpi['bid'], tpi['ask'], tpi['time']])
                arr3d = np.concatenate([arr0, arr1[:,::2]], axis=-1)
                market_second[pair] = {'ps': [tp], 'bs': [arr3d]}
            yield market_second


    @staticmethod
    async def gen_array_async(market_replay, markets, width_per_side=64, zoom_frac=1/256):
        #market_replay = self.multireplayL2(markets)
        prev_price = {p: None for p in markets}
        spacing = np.arange(width_per_side)
        #spacing = np.square(spacing) + spacing
        spacing = spacing / spacing[-1]
        spacing = np.arcsin(spacing)*3 - spacing*2
        async for second in market_replay:
            market_second = {}#collections.defaultdict(list)
            for pair in markets:
                sec = second[pair]
                prev_price[pair] = prev_price[pair] or sec['price']
                bib,aib,trb,tra  = BookShapper.bin_books(sec['bids'],sec['asks'],sec['trades'], ref_price=prev_price[pair], zoom_frac=zoom_frac, spacing=spacing)
                prev_price[pair] = sec['emaPrice']
                arr0 = bib.values - aib.values
                arr1 = tra.values - trb.values
                utc = datetime.datetime.utcfromtimestamp(sec['time'])
                d = sec['time'] // (3600 * 24) # int(utc.strftime('%y%m%d'))
                t = sec['time'] % (3600 * 24)  #float(utc.strftime('%H%M%S.%f'))
                tp = np.array([[sec['price'], sec['bids'].index[0], sec['asks'].index[0]], [d, t, 0]], dtype=np.float32)
                # print('tp', tp)
                arr3d = np.concatenate([arr0, arr1[:,::2]], axis=-1)
                market_second[pair] = {'ps': [tp], 'bs': [arr3d]}
            yield market_second


    @staticmethod
    def build_time_level_trade(books, prices, sidebips=32, sidesteps=64):
#        sidesteps = books.shape[1] // 2
        pricestep = prices[0, 0, 0] * 0.0001 * sidebips / sidesteps
        FUTURE = 120#0*10
        time2levels = np.zeros_like(books[:, :2*sidesteps, :1]) + FUTURE
        #########################################################
                                    #######
        for i in tqdm(range(prices.shape[0])):
                                    #######
        #########################################################
            timeupdn = []
            for j in range(sidesteps):
                thresh = j * pricestep
                [p, b, a], [d, t, _] = prices[i]
                waitUp = prices[i:i+FUTURE, 0, 1] < a + thresh
                waitDn = prices[i:i+FUTURE, 0, 2] > b - thresh
                timeUp = np.argmin(waitUp) or FUTURE*10
                timeDn = np.argmin(waitDn) or FUTURE*10
                timeupdn.insert(0, [timeDn])
                timeupdn.append([timeUp])
            time2levels[i] = timeupdn
        return time2levels.astype(np.float32)


    @staticmethod
    def build(total, element, reduce_func=None):
#        sidebips = 32
#        sidesteps = 64
        force_save = element is None
        element = element or total
        for market,second in element.items():
            dt = total[market]['ps'][-1][1] # datetime.datetime.strptime(str(int(total[market]['ps'][-1][1,0])), '%y%m%d').date()
            datetotal = datetime.datetime.utcfromtimestamp(int(dt[0]) * 3600 * 24 + int(dt[1])).date()
            de = element[market]['ps'][-1][1] # dateeleme = datetime.datetime.strptime(str(int(element[market]['ps'][-1][1,0])), '%y%m%d').date()
            # print(dt, dt[0] * 3600 * 24 + dt[1])
            dateeleme = datetime.datetime.utcfromtimestamp(int(de[0]) * 3600 * 24 + int(de[1])).date()
            newDay = datetotal < dateeleme
            if not (newDay or force_save):
                for name,arrs in second.items():
                    total[market][name] += arrs
            else:
                arrday_bs = np.stack(total[market]['bs']).astype(np.float32)
                np.save(f'data/{datetotal}-{market}-bs.npy', arrday_bs)
                arrday_ps = np.stack(total[market]['ps']).astype(np.float32)
                np.save(f'data/{datetotal}-{market}-ps.npy', arrday_ps)
                # np.save(f'data/{datetotal}-{market}-ptbs.npy', np.concatenate([arrday_ps, arrday_bs], axis=1))
                if reduce_func is not None:
                    t2l = reduce_func(
                        np.stack(total[market]['bs']).astype(np.float32),
                        np.stack(total[market]['ps']).astype(np.float32),
                    )
                    np.save(f'data/{datetotal}-{market}-time2level.npy', t2l)
            if newDay:
                for name,arrs in second.items():
                    total[market][name] = arrs
        return total

    @staticmethod
    def accumulate_array(genarr, markets):
        genacc = aioitertools.accumulate(genarr, BookShapper.build)
        return genacc



if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    shapper = loop.run_until_complete(BookShapper.create())
