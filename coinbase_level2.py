import asyncio
import httpx

async def get_level2_market_data(product_id="ETH-USD", top_n=5, interval=1):
    url = f"https://api.pro.coinbase.com/products/{product_id}/book?level=2"

    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(url)
                response.raise_for_status()  # Raises exception for 4XX/5XX responses
                data = response.json()

                # Clear the screen to make the data readable in a terminal
                print("\033[H\033[J", end="")
                print(f"Fetching Level 2 Market Data for {product_id} every {interval} seconds...")

                # Display the top bids and asks
                print("Top Bids:")
                for bid in data['bids'][:top_n]:
                    price, size = bid[0:2]
                    print(f"Price: {price}, Size: {size}")

                print("\nTop Asks:")
                for ask in data['asks'][:top_n]:
                    price, size = ask[0:2]
                    print(f"Price: {price}, Size: {size}")

                await asyncio.sleep(interval)

            except Exception as e:
                print(f"An error occurred: {e}")
                break  # Or handle the error as you see fit

async def fetch_recent_trades(product_id="ETH-USD", interval=1):
    """
    Fetches and displays the most recent trades for a given product ID from Coinbase Pro asynchronously.

    Args:
    - product_id (str): The ID of the product to fetch trades for.
    - interval (int): Interval in seconds between data fetches.
    """
    url = f"https://api.pro.coinbase.com/products/{product_id}/trades"

    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(url)
                response.raise_for_status()  # Raises exception for 4XX/5XX responses
                trades = response.json()

                # Clear the screen to make the data readable in a terminal
                print("\033[H\033[J", end="")
                print(f"Fetching Recent Trades for {product_id} every {interval} seconds...")
                print("Most Recent Trades:")

                # Display the most recent trades
                for trade in trades:
                    print(f"Time: {trade['time']}, Price: {trade['price']}, Size: {trade['size']}, Side: {trade['side']}")

                await asyncio.sleep(interval)

            except Exception as e:
                print(f"An error occurred: {e}")
                break  # Or handle the error as you see fit

if __name__ == "__main__":
    # asyncio.run(get_level2_market_data(product_id="ETH-USD", top_n=5, interval=1))
    asyncio.run(fetch_recent_trades(product_id="ETH-USD", interval=1))
