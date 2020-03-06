import asyncio
import unittest

from deep_orderbook.replayer import Replayer


class ReplayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(None)

        cls.replayer = Replayer(data_folder='../crypto-trading/data/L2')
        cls.symb = 'BTCUSDT'

    def test_01_create(self):
            upd = self.replayer.updates_files(pair=self.symb)
            end = upd[0].split('_')[-1]
            self.assertEqual(end, 'update.json')
