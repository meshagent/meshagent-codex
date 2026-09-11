"""Check a packaged Code Mode host using the pinned upstream stdio protocol."""

import argparse
import asyncio
import json
import struct
from pathlib import Path


async def exercise_host(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    async def send(message):
        payload = json.dumps(message).encode()
        writer.write(struct.pack("<I", len(payload)) + payload)
        await writer.drain()

    async def receive(message_type, request_id=None):
        while True:
            size = struct.unpack("<I", await reader.readexactly(4))[0]
            if size > 64 * 1024 * 1024:
                raise RuntimeError(f"Invalid host frame size: {size}")
            message = json.loads(await reader.readexactly(size))
            if message["type"] == "cell/closed":
                continue
            if message["type"] != message_type or (
                request_id is not None and message["id"] != request_id
            ):
                raise RuntimeError(f"Unexpected host response: {message}")
            if request_id is None:
                return message
            result = message["result"]
            if result["status"] != "ok":
                raise RuntimeError(f"Host operation failed: {result}")
            return result["value"]

    await send(
        {
            "type": "connection/hello",
            "supportedVersions": [1],
            "requiredCapabilities": [],
            "optionalCapabilities": [],
        }
    )
    hello = await receive("connection/ready")
    if hello["selectedVersion"] != 1:
        raise RuntimeError(f"Unexpected protocol version: {hello}")

    session_id = "distribution-smoke"
    for request_id, method in enumerate(
        ("session/open", "session/execute", "session/shutdown"), start=1
    ):
        request = {"method": method, "sessionId": session_id}
        if method == "session/execute":
            request["request"] = {
                "tool_call_id": "smoke",
                "enabled_tools": [],
                "source": 'text(6 * 7); text(new Intl.NumberFormat("en-US").format(1234567.89));',
                "yield_time_ms": 30_000,
                "max_output_tokens": 100,
            }
        await send({"type": "operation/request", "id": request_id, "request": request})
        response = await receive("operation/response", request_id)
        if method == "session/execute":
            if response["type"] != "execution/started":
                raise RuntimeError(f"Execution did not start: {response}")
            outcome = await receive("execute/initialResponse", request_id)
            result = outcome["Result"]
            expected = [
                {"type": "input_text", "text": "42"},
                {"type": "input_text", "text": "1,234,567.89"},
            ]
            if result["error_text"] is not None or result["content_items"] != expected:
                raise RuntimeError(f"JavaScript/ICU smoke test failed: {result}")
        else:
            expected_type = (
                "session/ready" if method == "session/open" else "session/closed"
            )
            if response != {"type": expected_type, "sessionId": session_id}:
                raise RuntimeError(f"Unexpected session response: {response}")


async def smoke(binary: Path):
    process = await asyncio.create_subprocess_exec(
        str(binary.resolve()),
        "--listen",
        "stdio",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(45):
            await exercise_host(process.stdout, process.stdin)
            process.stdin.close()
            if await process.wait() != 0:
                raise RuntimeError(f"Host exited with status {process.returncode}")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    print("Code Mode host passed: handshake, JavaScript, ICU, and session shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    asyncio.run(smoke(parser.parse_args().binary))
