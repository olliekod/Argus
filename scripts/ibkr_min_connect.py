import asyncio

from ib_insync import IB

from src.core.asyncio_compat import ensure_event_loop, run_sync


async def main() -> None:
    ensure_event_loop()
    ib = IB()
    ib.connect("127.0.0.1", 4002, clientId=42)
    print("Connected", ib.isConnected())
    print("ServerTime", ib.reqCurrentTime())
    ib.disconnect()


if __name__ == "__main__":
    run_sync(main())
