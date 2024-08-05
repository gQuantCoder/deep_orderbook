import polars as pl
import asyncio
import aiofiles
import os
import json
import pyinstrument
from tqdm.auto import tqdm


async def process_line(line, schema, categories):
    data = json.loads(line)
    event = data["events"][0]
    updates = event["updates"]

    # Create a DataFrame from the updates with predefined schema
    df = pl.DataFrame(updates, schema=schema)

    # Add common columns to the DataFrame with predefined categories
    df = df.with_columns(
        [
            pl.lit(data["channel"]).alias("channel"),
            pl.lit(data["timestamp"]).alias("timestamp"),
            pl.lit(data["sequence_num"]).alias("sequence_num"),
            pl.lit(event["type"]).alias("type"),
            pl.lit(event["product_id"]).alias("product_id"),
        ]
    )

    # Convert string columns to categorical types
    for col, cats in categories.items():
        df = df.with_columns(pl.col(col).cast(pl.Categorical).cat.set_ordering(cats))

    return df


async def read_jsonl_file(file_path, schema, categories):
    dataframes = []
    async with aiofiles.open(file_path, 'r') as file:
        async for line in tqdm(file, desc="Processing lines"):
            df = await process_line(line, schema, categories)
            dataframes.append(df)
    return pl.concat(dataframes, rechunk=True)


def polarize(input_file, explode=None):
    df = pl.read_ndjson(input_file)
    df = df.with_columns(
        [
            pl.col('channel').cast(
                pl.Enum(['l2_data', 'market_trades', 'subscriptions'])
            ),
            pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ"),
        ]
    )
    if explode:
        df = df.explode('events')
        df = df.unnest('events')
        if explode == 'updates':
            df = df.explode('updates')
            df = df.unnest('updates')
            df = df.with_columns(
                [
                    pl.col('type').cast(pl.Enum(['snapshot', 'update'])),
                    pl.col('product_id').cast(pl.Enum(['BTC-USD'])),
                    pl.col('side').cast(pl.Enum(['bid', 'offer'])),
                ]
            )
        if explode == 'trades':
            df = df.explode('trades')
            df = df.unnest('trades')
            df = df.with_columns(
                [
                    pl.col('type').cast(pl.Enum(['snapshot', 'update'])),
                    pl.col('product_id').cast(pl.Enum(['BTC-USD'])),
                    pl.col('side').cast(pl.Enum(['BUY', 'SELL'])),
                ]
            )
    return df

    # # Convert the DataFrame to a list of dictionaries
    # json_list = df_events.to_dict(as_series=False)


def depolarize(df, regroup=None):
    if regroup:
        if regroup == 'updates':
            # Reconstruct the 'updates' nested structure
            df = df.group_by(
                ['channel', 'timestamp', 'sequence_num', 'product_id', 'type'],
                maintain_order=True,
            ).agg([pl.struct(['price', 'size', 'side']).alias('updates')])

        if regroup == 'trades':
            # Reconstruct the 'trades' nested structure
            df = df.group_by(
                ['channel', 'timestamp', 'sequence_num', 'product_id', 'type'],
                maintain_order=True,
            ).agg([pl.struct(['price', 'size', 'side']).alias('trades')])

        # Reconstruct the 'events' nested structure
        df = df.group_by(
            ['channel', 'timestamp', 'sequence_num'],
            maintain_order=True,
        ).agg([pl.struct(['type', 'product_id', 'updates']).alias('events')])

    return df


async def main(folder='data/L2/BTC-USD/'):
    input_file = os.path.join(folder, '2024-08-04T23-00-00_update.jsonl')
    input_file = os.path.join(folder, '2024-08-05T04-26-38_update.jsonl')

    output_file = input_file.replace('.jsonl', '.parquet')

    EXPLODE = None
    # EXPLODE = 'updates'

    df = polarize(input_file, explode=EXPLODE)

    print(df)
    df.write_parquet(output_file)
    print(f"Data has been written to {output_file}")

    with pyinstrument.Profiler() as profiler:
        df_read = pl.read_parquet(output_file)
        df_read = depolarize(df_read, regroup=EXPLODE)
        for dd in tqdm(df_read.iter_rows(named=True)):
            pass
        print(dd)
        
    profiler.open_in_browser(timeline=True)

    assert df_read.equals(df)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
