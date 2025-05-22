"""
modified from https://modelcontextprotocol.io/quickstart/client  in tab 'python'
"""

import asyncio
from typing import Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from rich import print as rprint

from dotenv import load_dotenv

from augmented.mcp_tools import PresetMcpTools
from augmented.utils.info import PROJECT_ROOT_DIR
from augmented.utils.pretty import RICH_CONSOLE

load_dotenv()


class MCPClient:
    def __init__(
        self,
        name: str,
        command: Optional[str] = None, # Made command optional
        args: Optional[list[str]] = None, # Made args optional
        version: str = "0.0.1",
        url: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.name = name
        self.version = version
        self.command = command
        self.args = args
        self.url = url
        self.token = token
        self.tools: list[Tool] = []

    async def init(self) -> None:
        await self._connect_to_server()

    async def cleanup(self) -> None:
        try:
            await self.exit_stack.aclose()
        except Exception:
            rprint("Error during MCP client cleanup, traceback and continue!")
            RICH_CONSOLE.print_exception()

    def get_tools(self) -> list[Tool]:
        return self.tools

    async def _connect_to_server(
        self,
    ) -> None:
        """
        Connect to an MCP server
        """
        if self.url:
            # Remote connection using streamablehttp_client
            # TODO: Investigate proper token authentication for streamablehttp_client
            # For now, attempting connection without specific auth mechanism if not obvious
            # headers = {}
            # if self.token:
            #     headers["Authorization"] = f"Bearer {self.token}"

            # Assuming streamablehttp_client takes url and optional headers
            # The exact signature for passing token/auth needs to be confirmed
            # from mcp.client.streamable_http documentation.
            read_stream, write_stream, _response_headers = await self.exit_stack.enter_async_context(
                streamablehttp_client(self.url) # Add headers=headers if supported
            )
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
        else:
            # Local connection using stdio_client
            if not self.command:
                raise ValueError("Command must be provided for stdio connections.")
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args if self.args else [],
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
    for mcp_tool in [
        PresetMcpTools.filesystem.append_mcp_params(f" {PROJECT_ROOT_DIR!s}"),
        PresetMcpTools.fetch,
    ]:
        rprint(mcp_tool.shell_cmd)
        mcp_client = MCPClient(**mcp_tool.to_common_params())
        await mcp_client.init()
        tools = mcp_client.get_tools()
        rprint(tools)
        await mcp_client.cleanup()


if __name__ == "__main__":
    asyncio.run(example())
