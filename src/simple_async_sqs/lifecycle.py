import asyncio
import logging
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncContextManager, AsyncIterator, Callable, Protocol


if TYPE_CHECKING:
    from types_aiobotocore_sqs.type_defs import MessageTypeDef

from .queue_client import QueueClient

logger = logging.getLogger(__name__)


class LifeCycle(Protocol):
    """Defines the lifecycle of a single message.

    That is what happens
        - when a message is processed successfully
        - when a message errors
        - during the message processing
    """

    def __call__(
        self, message: MessageTypeDef
    ) -> AsyncContextManager[MessageTypeDef]: ...


@dataclass
class RetryLifeCycle(LifeCycle):
    """Each messge will be retried after `retry_interval` seconds forever.

    The number of retries can be set on the DLQ configuration in SQS.
    """

    client: QueueClient
    retry_interval: int

    @asynccontextmanager
    async def __call__(self, message: MessageTypeDef) -> AsyncIterator[MessageTypeDef]:
        try:
            yield message
        except Exception:
            logger.exception(
                "Exception whilst processing message, retrying in %s",
                self.retry_interval,
            )
            await self.client.nack(message, self.retry_interval)
        else:
            await self.client.ack(message)


@dataclass
class ExponentialRetryLifeCycle(LifeCycle):
    """Each messge will be retried inititally for `retry_interval` seconds backing off exponentially.

    The number of retries can be set on the DLQ configuration in SQS.
    """

    client: QueueClient
    retry_interval: int

    @asynccontextmanager
    async def __call__(self, message: MessageTypeDef) -> AsyncIterator[MessageTypeDef]:
        try:
            yield message
        except Exception:
            logger.exception(
                "Exception whilst processing message, retrying in %s",
                self.retry_interval,
            )
            receive_count = int(
                message.get("Attributes", {}).get("ApproximateReceiveCount", 1)
            )
            await self.client.nack(
                message, retry_timeout=self.retry_interval * 2**receive_count
            )
        else:
            await self.client.ack(message)


@dataclass
class HeartbeatLifeCycle(LifeCycle):
    """Keeps the message alive by sending heartbeats every `interval` seconds

    An optional inner life cycle can be provided to define what happens on success or error.
    """

    client: QueueClient
    interval: int
    inner_life_cycle: Callable[
        [MessageTypeDef], AsyncContextManager[MessageTypeDef]
    ] = nullcontext

    @asynccontextmanager
    async def auto_keep_alive(self, message: MessageTypeDef) -> AsyncIterator[None]:
        async def _keep_alive():
            while True:
                await asyncio.sleep(self.interval - 1)
                await self.client.heartbeat(message, self.interval)

        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(_keep_alive())
            try:
                yield
            finally:
                task.cancel()

    @asynccontextmanager
    async def __call__(self, message: MessageTypeDef) -> AsyncIterator[MessageTypeDef]:
        async with self.auto_keep_alive(message):
            async with self.inner_life_cycle(message) as message:
                yield message
