import asyncio

from simple_async_sqs.queue_client import QueueClient


async def process(client: QueueClient):
    async for message in client.consume():
        try:
            print(message.get("Body"))
            ...
        except Exception:
            await client.nack(message, retry_timeout=20)
        else:
            await client.ack(message)


async def process_single():
    async with QueueClient.create("my-queue") as client:
        await process(client)


async def process_parallel(workers: int):
    async with QueueClient.create("my-queue") as client:
        async with asyncio.TaskGroup() as tg:
            for _ in range(workers):
                tg.create_task(process(client))