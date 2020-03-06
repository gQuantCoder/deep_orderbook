import asyncio
import unittest

from deep_orderbook.recorder import Receiver


class ReceiverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(None)

        cls.symb = 'BTCUSDT'
        async def go():
            cls.receiver = await Receiver.create(markets=[cls.symb])
        cls.loop.run_until_complete(go())

    @classmethod
    def tearDownClass(cls):
        async def go():
            await cls.receiver.bm.close()
            await asyncio.sleep(1)
        cls.loop.run_until_complete(go())
        del cls.receiver

    def test_01_create(self):
            k = list(self.receiver.depth_managers.keys())[0]
            self.assertEqual(k, self.symb)

    def test_02_connection(self):
        async def go():
            for i in range(100):
                trades = self.receiver.trade_managers[self.symb]
                if len(trades):
                    self.assertTrue(True)
                    return
            self.assertTrue(False)
        self.loop.run_until_complete(go())
