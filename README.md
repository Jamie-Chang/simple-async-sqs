# simple-async-sqs
`simple-async-sqs` is a opinionated minimalistic async Python client to interact with SQS. 

`simple-async-sqs` is for developers who are tired of repeated configurations in task frameworks and prefer a simple message processing library. 


## Installation

```bash
uv add simple-async-sqs
```

Optionally add the type stubs for used in development:

```bash
uv add --dev simple-async-sqs[stubs]
```

## Usage
A consumer can be simply created by `Client.consume` which is simply an `AsyncIterator`.

Messages must be either ack'd or nack'd after processing.
```py
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
```

We can also easily parallelise the work here using `asyncio.TaskGroup`:

```py
async def process_parallel(workers: int):
    async with QueueClient.create("my-queue") as client:
        async with asyncio.TaskGroup() as tg:
            for _ in range(workers):
                tg.create_task(process(client))
```


### Lifecycles

Lifecycles define what happens during message processing - on success, error, and during processing. They provide a way to handle retries, heartbeats, and other message lifecycle concerns automatically.

#### RetryLifeCycle

Retries failed messages after a fixed interval indefinitely. The number of retries can be configured via SQS DLQ settings.

```py
from simple_async_sqs.lifecycle import RetryLifeCycle

async def process_with_retry():
    async with QueueClient.create("my-queue") as client:
        lifecycle = RetryLifeCycle(client, retry_interval=30)
        async for message in client.consume():
            async with lifecycle(message) as msg:
                # Process message
                print(msg.get("Body"))
                # On exception: automatically retries after 30 seconds
                # On success: automatically acks the message
```

#### ExponentialRetryLifeCycle

Retries failed messages with exponential backoff. Each retry doubles the wait time based on the message's receive count.

```py
from simple_async_sqs.lifecycle import ExponentialRetryLifeCycle

async def process_with_exponential_retry():
    async with QueueClient.create("my-queue") as client:
        lifecycle = ExponentialRetryLifeCycle(client, retry_interval=10)
        async for message in client.consume():
            async with lifecycle(message) as msg:
                # Process message
                print(msg.get("Body"))
                # On exception: retries with exponential backoff (10s, 20s, 40s, etc.)
                # On success: automatically acks the message
```

#### HeartbeatLifeCycle

Keeps messages alive by extending their visibility timeout with periodic heartbeats. Useful for long-running message processing.

```py
from simple_async_sqs.lifecycle import HeartbeatLifeCycle, RetryLifeCycle

async def process_with_heartbeat():
    async with QueueClient.create("my-queue") as client:
        # Heartbeat every 60 seconds, with retry on failure
        lifecycle = HeartbeatLifeCycle(
            client, 
            interval=60, 
            inner_life_cycle=RetryLifeCycle(client, retry_interval=30)
        )
        async for message in client.consume():
            async with lifecycle(message) as msg:
                # Message visibility extended automatically every 60 seconds
                await asyncio.sleep(300)  # Long processing
                print(msg.get("Body"))
                # On exception: retries after 30 seconds
                # On success: automatically acks the message
```

### Producer
We can simply produce a message by:
```py
await client.producer("my_message_payload", delay=10)
```