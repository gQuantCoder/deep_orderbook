import tensorflow as tf
import numpy as np
import functools

from config import args
from deep_orderbook import replayer, shapper

SYMBOL = args.symbol

MAX_DAYS_FOR_NOW = 2

def alpha(arr_time2level):
    return 10 / (1 + (arr_time2level))


class DataFeed(replayer.Replayer):
    def __init__(self, data_folder, side_bips, side_width, date_regexp=''):
        self.replay = replayer.Replayer('../data/crypto', date_regexp=args.date_regexp)
        file_gen = self.replay.training_files(SYMBOL, side_bips=side_bips, side_width=side_width)
        fn_bs, fn_ps, fn_ts = zip(*list(file_gen)[-1:])
        arr_books = np.concatenate(list(map(np.load, fn_bs)))
        arr_prices = np.concatenate(list(map(np.load, fn_ps)))
        arr_time2level = np.concatenate(list(map(np.load, fn_ts)))
        print('time2level', arr_time2level.shape, arr_time2level.min(), arr_time2level.mean(), arr_time2level.max())

        self.side_bips = side_bips
        self.side_width = side_width
        self.widthbooks = side_width * 2
        self.sidesteps = arr_books.shape[1] // 2
        self.chanbooks = arr_books.shape[2]

        assert side_width == self.sidesteps

        self.sample_arr_books = self.batch_length(arr_books, 2048)
        self.sample_arr_prices = self.batch_length(arr_prices, 2048)
        self.sample_arr_time2level = self.batch_length(arr_time2level, 2048)

        self.sample_arr_alpha = alpha(self.sample_arr_time2level)

    def raw_numpy_gen(self, pair, frac_from=0.0, frac_to=1.0):
        file_names = list(self.replay.training_files(pair=pair, side_bips=self.side_bips, side_width=self.side_width))
        num = len(file_names)
        rangefrom = int(num * frac_from)
        rangeto = int(num * frac_to)
        for fn_bs, fn_ps, fn_ts in file_names[rangefrom:rangeto]:
            print(fn_bs, fn_ps, fn_ts)
            arr_books, arr_prices, arr_time2level = np.load(fn_bs), np.load(fn_ts), np.load(fn_ps)
            assert arr_books.shape[0] ==  arr_prices.shape[0]
            assert arr_books.shape[0] ==  arr_time2level.shape[0]
            yield arr_books, arr_prices, arr_time2level

    def batch_length(self, arr, sample_length):
        sample_num = arr.shape[0] // sample_length
        arr_length = arr[:sample_num*sample_length]
        arr_length = arr_length.reshape([-1, sample_length] + list(arr.shape[1:]))
        return arr_length

    def data_flow(self, split, batch_size, sample_length):
        assert len(split) == 1
        dataset = tf.data.Dataset.from_generator(
                        functools.partial(self.raw_numpy_gen, frac_to=split[0]), 
                        (tf.float32, tf.float32, tf.float32),
                        (tf.TensorShape([None, self.widthbooks, self.chanbooks]),
                         tf.TensorShape([None, self.widthbooks, 1]),
                         tf.TensorShape([None, 2, 3])))
        
        arr_books = self.sample_arr_books
        arr_prices = self.sample_arr_prices
        arr_time2level = self.sample_arr_time2level
        arr_alpha = alpha(arr_time2level)
        SAMPLES_TRAIN = arr_books.shape[0] // 2 + arr_books.shape[0] % 2
        SAMPLES_VALID = arr_books.shape[0] // 2
        
        if args.samples_per_epoch == 0:
            args.samples_per_epoch = SAMPLES_TRAIN

        train_ds = tf.data.Dataset.from_tensor_slices(
            (arr_books[:SAMPLES_TRAIN], arr_alpha[:SAMPLES_TRAIN], arr_prices[:SAMPLES_TRAIN]))
        valid_ds = tf.data.Dataset.from_tensor_slices(
            (arr_books[-SAMPLES_VALID:], arr_alpha[-SAMPLES_VALID:], arr_prices[-SAMPLES_VALID:]))

        if args.seed:
            train_ds = train_ds.shuffle(args.samples_per_epoch, seed=args.seed)
            valid_ds = valid_ds.shuffle(args.samples_per_epoch, seed=args.seed)

        train_ds = train_ds.batch(batch_size)
        valid_ds = valid_ds.batch(batch_size)

        print("train_ds", SAMPLES_TRAIN, train_ds)
        print("valid_ds", SAMPLES_VALID, valid_ds)

        return train_ds, valid_ds

if False:
    SAMPLE_LENGTH = args.sample_length
    BATCH_SIZE = args.batch_size

    # neeed this ?    https://gist.github.com/seberg/3866040
    ######################################################
    ######################################################
    SAMPLE_NUM = arr_books.shape[0] // SAMPLE_LENGTH
    print(SAMPLE_NUM, "samples of length", SAMPLE_LENGTH)
    arr_books = arr_books[:SAMPLE_NUM*SAMPLE_LENGTH]
    arr_books = arr_books.reshape([-1, SAMPLE_LENGTH, sidesteps * 2, chanbooks])
    arr_alpha = arr_alpha[:SAMPLE_NUM*SAMPLE_LENGTH]
    arr_alpha = arr_alpha.reshape([-1, SAMPLE_LENGTH] + list(arr_alpha.shape[1:]))
    arr_prices = arr_prices[:SAMPLE_NUM*SAMPLE_LENGTH]
    arr_prices = arr_prices.reshape([-1, SAMPLE_LENGTH] + list(arr_prices.shape[1:]))
    ######################################################
    ######################################################
    print('arr_books', arr_books.shape, arr_books.min(), arr_books.mean(), arr_books.max())
    print('arr_prices', arr_prices.shape, arr_prices.min(), arr_prices.mean(), arr_prices.max())
    print('alpha', arr_alpha.shape, arr_alpha.min(), arr_alpha.mean(), arr_alpha.max())

    print('arr_books', arr_books.shape, 'arr_alpha', arr_alpha.shape)

    SAMPLES_TRAIN = arr_books.shape[0] // 2 + arr_books.shape[0] % 2
    SAMPLES_VALID = arr_books.shape[0] // 2

    if args.samples_per_epoch == 0:
        args.samples_per_epoch = SAMPLES_TRAIN

    train_ds = tf.data.Dataset.from_tensor_slices(
        (arr_books[:SAMPLES_TRAIN], arr_alpha[:SAMPLES_TRAIN], arr_prices[:SAMPLES_TRAIN]))
    valid_ds = tf.data.Dataset.from_tensor_slices(
        (arr_books[-SAMPLES_VALID:], arr_alpha[-SAMPLES_VALID:], arr_prices[-SAMPLES_VALID:]))

    if args.seed:
        train_ds = train_ds.shuffle(args.samples_per_epoch, seed=args.seed)
        valid_ds = valid_ds.shuffle(args.samples_per_epoch, seed=args.seed)

    train_ds = train_ds.batch(BATCH_SIZE)
    valid_ds = valid_ds.batch(BATCH_SIZE)

    print("train_ds", SAMPLES_TRAIN, train_ds)
    print("valid_ds", SAMPLES_VALID, valid_ds)

