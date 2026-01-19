import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from simple_async_sqs.lifecycle import (
    ExponentialRetryLifeCycle,
    HeartbeatLifeCycle,
    RetryLifeCycle,
)


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def sample_message():
    return {
        "MessageId": "123",
        "ReceiptHandle": "receipt-123",
        "Body": "test message",
        "Attributes": {"ApproximateReceiveCount": "1"},
    }


@pytest.mark.asyncio
async def test_retry_lifecycle_success(mock_client, sample_message):
    lifecycle = RetryLifeCycle(client=mock_client, retry_interval=5)

    async with lifecycle(sample_message) as msg:
        assert msg == sample_message

    mock_client.ack.assert_called_once_with(sample_message)
    mock_client.nack.assert_not_called()


@pytest.mark.asyncio
async def test_retry_lifecycle_exception(mock_client, sample_message):
    lifecycle = RetryLifeCycle(client=mock_client, retry_interval=5)

    async with lifecycle(sample_message) as msg:
        assert msg == sample_message
        raise ValueError("test exception")

    mock_client.nack.assert_called_once_with(sample_message, 5)
    mock_client.ack.assert_not_called()


@pytest.mark.asyncio
async def test_exponential_retry_lifecycle_success(mock_client, sample_message):
    lifecycle = ExponentialRetryLifeCycle(client=mock_client, retry_interval=2)

    async with lifecycle(sample_message) as msg:
        assert msg == sample_message

    mock_client.ack.assert_called_once_with(sample_message)
    mock_client.nack.assert_not_called()


@pytest.mark.asyncio
async def test_exponential_retry_lifecycle_exception_first_retry(mock_client, sample_message):
    lifecycle = ExponentialRetryLifeCycle(client=mock_client, retry_interval=2)

    async with lifecycle(sample_message) as msg:
        assert msg == sample_message
        raise ValueError("test exception")

    # First retry: 2 * 2^1 = 4
    mock_client.nack.assert_called_once_with(sample_message, retry_timeout=4)
    mock_client.ack.assert_not_called()


@pytest.mark.asyncio
async def test_exponential_retry_lifecycle_exception_multiple_retries(mock_client):
    lifecycle = ExponentialRetryLifeCycle(client=mock_client, retry_interval=2)

    # Second retry
    message: Any = {
        "MessageId": "123",
        "ReceiptHandle": "receipt-123",
        "Body": "test message",
        "Attributes": {"ApproximateReceiveCount": "2"},
    }

    async with lifecycle(message) as msg:
        assert msg == message
        raise ValueError("test exception")

    # Second retry: 2 * 2^2 = 8
    mock_client.nack.assert_called_once_with(message, retry_timeout=8)
    mock_client.ack.assert_not_called()


@pytest.mark.asyncio
async def test_exponential_retry_lifecycle_no_receive_count(mock_client):
    lifecycle = ExponentialRetryLifeCycle(client=mock_client, retry_interval=2)

    message: Any = {
        "MessageId": "123",
        "ReceiptHandle": "receipt-123",
        "Body": "test message",
        # No Attributes
    }

    async with lifecycle(message) as msg:
        assert msg == message
        raise ValueError("test exception")

    # No receive count defaults to 1: 2 * 2^1 = 4
    mock_client.nack.assert_called_once_with(message, retry_timeout=4)
    mock_client.ack.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_lifecycle_success(mock_client, sample_message):
    lifecycle = HeartbeatLifeCycle(client=mock_client, interval=1)  # Reduce interval for test

    async with lifecycle(sample_message) as msg:
        assert msg == sample_message
        await asyncio.sleep(1.5)  # Wait for heartbeat to trigger

    mock_client.heartbeat.assert_called()
    mock_client.ack.assert_not_called()
    mock_client.nack.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_lifecycle_with_inner_lifecycle(mock_client, sample_message):
    inner_lifecycle = MagicMock()
    inner_lifecycle.return_value.__aenter__ = AsyncMock(return_value=sample_message)
    inner_lifecycle.return_value.__aexit__ = AsyncMock(return_value=None)

    lifecycle = HeartbeatLifeCycle(
        client=mock_client, interval=1, inner_life_cycle=inner_lifecycle
    )

    async with lifecycle(sample_message) as msg:
        assert msg == sample_message
        await asyncio.sleep(1.5)  # Wait for heartbeat

    mock_client.heartbeat.assert_called()
    inner_lifecycle.assert_called_once_with(sample_message)


@pytest.mark.asyncio
async def test_heartbeat_lifecycle_exception_with_inner(mock_client, sample_message):
    inner_lifecycle = MagicMock()
    inner_lifecycle.return_value.__aenter__ = AsyncMock(return_value=sample_message)
    inner_lifecycle.return_value.__aexit__ = AsyncMock(return_value=None)

    lifecycle = HeartbeatLifeCycle(
        client=mock_client, interval=1, inner_life_cycle=inner_lifecycle
    )

    with pytest.raises(ExceptionGroup):
        async with lifecycle(sample_message) as msg:
            assert msg == sample_message
            await asyncio.sleep(1.5)  # Wait for heartbeat
            raise ValueError("test exception")

    mock_client.heartbeat.assert_called()
    inner_lifecycle.assert_called_once_with(sample_message)