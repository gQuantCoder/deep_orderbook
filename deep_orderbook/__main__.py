import argparse
import asyncio
from importlib.metadata import PackageNotFoundError, version


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='deepbook',
        description='Deep OrderBook CLI',
    )
    parser.add_argument('--version', action='store_true', help='Show version and exit')
    parser.add_argument(
        'command',
        nargs='?',
        choices=['record', 'replay'],
        help='record: live recorder, replay: parquet replay demo',
    )
    args = parser.parse_args()

    if args.version:
        try:
            print(version('deep-orderbook'))
        except PackageNotFoundError:
            print('unknown')
        return

    if args.command == 'record':
        from deep_orderbook.consumers.recorder import main as recorder_main

        asyncio.run(recorder_main())
        return

    if args.command == 'replay':
        from deep_orderbook.replayer import main as replay_main

        asyncio.run(replay_main())
        return

    parser.print_help()


if __name__ == '__main__':
    main()
