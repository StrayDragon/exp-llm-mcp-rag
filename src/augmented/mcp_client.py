"""
modified from https://modelcontextprotocol.io/quickstart/client  in tab 'python'
"""

import asyncio
import shlex
from typing import Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client

from rich import print as rprint

from dotenv import load_dotenv

from augmented.utils.info import PROJECT_ROOT_DIR
from augmented.utils.pretty import log_title

load_dotenv()


class MCPClient:
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        version: str = "0.0.1",
    ) -> None:
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.name = name
        self.version = version
        self.command = command
        self.args = args
        self.tools: list[Tool] = []

    async def init(self) -> None:
        await self._connect_to_server()

    async def close(self) -> None:
        await self.exit_stack.aclose()

    def get_tools(self) -> list[Tool]:
        return self.tools

    async def _connect_to_server(
        self,
    ) -> None:
        """
        Connect to an MCP server
        """
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params),
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        rprint("\nConnected to server with tools:", [tool.name for tool in self.tools])

    async def call_tool(self, name: str, params: dict[str, Any]):
        return await self.session.call_tool(name, params)


async def example() -> None:
    for mcp_name, cmd in [
        (
            "filesystem",
            f"npx -y @modelcontextprotocol/server-filesystem {PROJECT_ROOT_DIR!s}",
        ),
        (
            "fetch",
            "uvx mcp-server-fetch",
        ),
    ]:
        log_title(mcp_name)
        command, *args = shlex.split(cmd)
        mcp_client = MCPClient(
            name=mcp_name,
            command=command,
            args=args,
        )
        await mcp_client.init()
        tools = mcp_client.get_tools()
        rprint(tools)
        await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(example())
