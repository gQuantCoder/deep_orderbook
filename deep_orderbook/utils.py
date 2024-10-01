import logging


class Config:
    LOG_LEVEL = logging.WARNING
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    RUN_WHERE = 'local'

conf = Config()

class NoNewLineFileHandler(logging.FileHandler):
    terminator = ''

def make_handlers(filename='./replayer.log', filename_if_no_permission='./player.log'):
    mode = 'a'
    encoding = 'utf-8'
    try:
        main_h = logging.FileHandler(filename, mode, encoding=encoding)
        stream_h = NoNewLineFileHandler(filename, mode, encoding=encoding)
    except PermissionError:
        main_h = logging.FileHandler(filename_if_no_permission, mode, encoding=encoding)
        stream_h = NoNewLineFileHandler(
            filename_if_no_permission, mode, encoding=encoding
        )

    datefmt = '%H:%M:%S'
    fmt = f'''%(asctime)s.%(msecs)03d %(levelname)s {conf.RUN_WHERE} %(module)s - %(funcName)s: %(message)s'''
    formatter = logging.Formatter(fmt, datefmt, '%')
    main_h.setFormatter(formatter)

    no_newline_formatter = logging.Formatter('%(message)s', '', '%')
    stream_h.setFormatter(no_newline_formatter)

    return main_h, stream_h



def make_logger(name='deep_orderbook', level=logging.INFO):
    line_handler, noline_handler = make_handlers()
    logger = logging.getLogger(name)
    logger.addHandler(line_handler)
    logger.setLevel(level)

    return logger


logger = make_logger(level=conf.LOG_LEVEL)

