import asyncio
from dataclasses import dataclass, asdict
import json
import os
import yaml
from typing import Optional, Any, Dict

from rich import print as rprint

from augmented.chat_openai import AsyncChatOpenAI
from augmented.mcp_client import MCPClient
from augmented.mcp_tools import PresetMcpTools, McpToolInfo
from augmented.utils import pretty
from augmented.utils.info import DEFAULT_MODEL_NAME, PROJECT_ROOT_DIR

PRETTY_LOGGER = pretty.ALogger("[Agent]")


@dataclass
class Agent:
    model: str
    llm: AsyncChatOpenAI | None = None
    system_prompt: str = ""
    context: str = ""
    mcp_clients: list[MCPClient] = dataclasses.field(default_factory=list)
    current_mcp_configs_from_yaml: list[dict] = dataclasses.field(default_factory=list)


    async def init(self) -> None:
        PRETTY_LOGGER.title("INIT MCP CLIENTS & LLM TOOLS")
        self.mcp_clients = [] # Clear existing clients

        for server_config in self.current_mcp_configs_from_yaml:
            mcp_type = server_config.get("type")
            client: Optional[MCPClient] = None

            if mcp_type == "remote":
                url = server_config.get("url")
                if not url:
                    rprint(f"Warning: Remote MCP config missing 'url'. Skipping: {server_config}")
                    continue
                
                token = None
                if server_config.get("token_env"):
                    token = os.environ.get(server_config["token_env"])
                elif server_config.get("token"):
                    token = server_config["token"]
                
                client_name = server_config.get("name", f"remote-{url}")
                client = MCPClient(name=client_name, url=url, token=token)

            elif mcp_type == "local":
                client_name = server_config.get("name", "local_mcp_client")
                if server_config.get("preset_ref"):
                    preset_name = server_config["preset_ref"]
                    base_tool_info: Optional[McpToolInfo] = getattr(PresetMcpTools, preset_name, None)

                    if base_tool_info:
                        params_to_append = server_config.get("preset_mcp_params_append")
                        if params_to_append:
                            # Create a new McpToolInfo instance to avoid modifying the global preset
                            # This assumes McpToolInfo is a dataclass or has accessible attributes for copying
                            final_mcp_params = base_tool_info.mcp_params
                            if base_tool_info.mcp_params and params_to_append: # Ensure not appending to empty string if base is empty
                                final_mcp_params += " " + params_to_append.strip()
                            elif params_to_append:
                                final_mcp_params = params_to_append.strip()
                            
                            # Create a new instance using attributes from the base and modified params
                            # Need to ensure shell_cmd_pattern is correctly used or command/args are derived
                            # McpToolInfo.to_common_params() splits the shell_cmd. If we modify mcp_params,
                            # the shell_cmd must be reconstructed before calling to_common_params().
                            
                            # Reconstruct shell_cmd_pattern with new mcp_params before splitting
                            temp_tool_info = McpToolInfo(
                                name=base_tool_info.name, # Name for the tool info itself, not MCPClient
                                shell_cmd_pattern=base_tool_info.shell_cmd_pattern,
                                main_cmd_options=base_tool_info.main_cmd_options,
                                mcp_params=final_mcp_params
                            )
                            client_params = temp_tool_info.to_common_params()
                            client = MCPClient(name=client_name, **client_params)
                        else:
                            client = MCPClient(name=client_name, **base_tool_info.to_common_params())
                    else:
                        rprint(f"Warning: PresetMcpTool '{preset_name}' not found. Skipping: {server_config}")
                        continue
                else: # Explicit local command
                    command = server_config.get("command")
                    if not command:
                        rprint(f"Warning: Local MCP config missing 'command' (and not using 'preset_ref'). Skipping: {server_config}")
                        continue
                    args = server_config.get("args", [])
                    client = MCPClient(name=client_name, command=command, args=args)
            
            else:
                rprint(f"Warning: Unknown MCP type '{mcp_type}'. Skipping: {server_config}")
                continue

            if client:
                self.mcp_clients.append(client)

        # Initialize all configured MCP clients and collect their tools
        tools = []
        for mcp_client in self.mcp_clients:
            await mcp_client.init() # This connects the client
            tools.extend(mcp_client.get_tools())
        
        self.llm = AsyncChatOpenAI(
            self.model,
            tools=tools,
            system_prompt=self.system_prompt,
            context=self.context,
        )

    async def cleanup(self) -> None:
        PRETTY_LOGGER.title("CLEANUP MCP CLIENTS & LLM")

        # Cleanup MCP Clients
        # Use a copy for iteration if mcp_client.cleanup() modifies self.mcp_clients
        # (pop in original version did this, direct iteration should be fine if cleanup doesn't modify list)
        for mcp_client in self.mcp_clients:
            try:
                await mcp_client.cleanup()
            except Exception as e:
                rprint(f"Error during MCP client cleanup for {mcp_client.name}: {e!s}")
                # Optionally, log the full exception with RICH_CONSOLE.print_exception()
        self.mcp_clients = [] # Clear the list after cleanup

        # Cleanup LLM (if it has any specific cleanup)
        if self.llm and hasattr(self.llm, 'cleanup'): # Hypothetical cleanup for LLM
             await self.llm.cleanup()
        self.llm = None


    async def invoke(self, prompt_yaml_path: str) -> str | None:
        PRETTY_LOGGER.info(f"Invoking agent with YAML: {prompt_yaml_path}")
        try:
            with open(prompt_yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
        except Exception as e:
            rprint(f"Error loading or parsing YAML file {prompt_yaml_path}: {e!s}")
            return None

        prompt_text = yaml_content.get("prompt_text")
        if not prompt_text:
            rprint(f"Error: 'prompt_text' not found in YAML file {prompt_yaml_path}")
            return None

        self.current_mcp_configs_from_yaml = yaml_content.get("mcp_servers", [])
        self.system_prompt = yaml_content.get("system_prompt", self.system_prompt) # Allow overriding system_prompt from YAML
        self.context = yaml_content.get("context", self.context) # Allow overriding context from YAML


        await self.init() # Initialize MCP clients and LLM based on YAML
        
        return await self._invoke(prompt_text)

    async def _invoke(self, prompt: str) -> str | None:
        if self.llm is None:
            # This case should ideally be prevented by self.init() being called in invoke()
            rprint("Error: LLM not initialized. Call init() before invoke.")
            raise ValueError("LLM not initialized. Call init() before invoke.")
        chat_resp = await self.llm.chat(prompt)
        i = 0
        while True:
            PRETTY_LOGGER.title(f"INVOKE CYCLE {i}")
            i += 1
            # Process tool calls
            rprint(chat_resp) # Log the response, including potential tool calls
            if chat_resp.tool_calls:
                tool_messages_for_llm = []
                for tool_call in chat_resp.tool_calls:
                    target_mcp_client: Optional[MCPClient] = None
                    # Find the MCP client that provides this tool
                    for mcp_client in self.mcp_clients:
                        if tool_call.function.name in [
                            t.name for t in mcp_client.get_tools()
                        ]:
                            target_mcp_client = mcp_client
                            break
                    
                    tool_result_content = "Error: Tool not found"
                    if target_mcp_client:
                        PRETTY_LOGGER.title(f"TOOL USE `{tool_call.function.name}` on client `{target_mcp_client.name}`")
                        rprint("with args:", tool_call.function.arguments)
                        try:
                            # Arguments are expected to be a JSON string by the tool_call spec,
                            # but MCPClient.call_tool expects a dict.
                            tool_params = json.loads(tool_call.function.arguments)
                            mcp_result = await target_mcp_client.call_tool(
                                tool_call.function.name,
                                tool_params,
                            )
                            rprint("call result:", mcp_result)
                            # Assuming mcp_result has a method to get a JSON string suitable for the LLM
                            # or that model_dump_json() is appropriate.
                            tool_result_content = mcp_result.model_dump_json() 
                        except json.JSONDecodeError:
                            tool_result_content = "Error: Invalid JSON arguments for tool."
                            rprint(tool_result_content)
                        except Exception as e:
                            tool_result_content = f"Error executing tool: {e!s}"
                            rprint(tool_result_content)
                    else:
                         rprint(f"Warning: Tool '{tool_call.function.name}' not found in any MCP client.")
                    
                    # self.llm.append_tool_result(tool_call.id, tool_result_content) # Old method
                    # For OpenAI, the role should be 'tool' and content is the result, tool_call_id is needed.
                    tool_messages_for_llm.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name, # Recommended by OpenAI
                            "content": tool_result_content,
                        }
                    )

                # Send all tool results back to the LLM in one go
                if tool_messages_for_llm:
                    chat_resp = await self.llm.chat_with_tool_results(messages=chat_resp.messages + tool_messages_for_llm)
                else: # Should not happen if chat_resp.tool_calls was not empty
                    chat_resp = await self.llm.chat()

            else: # No tool calls, LLM provided a direct answer
                return chat_resp.content


# async def example() -> None:
#     # This example needs to be updated to use YAML input.
#     # A new example demonstrating YAML usage will be provided separately.
#     # For now, commenting this out.
#     # enabled_mcp_clients = []
#     # agent = None
#     # try:
#     #     for mcp_tool in [
#     #         PresetMcpTools.filesystem.append_mcp_params(f" {PROJECT_ROOT_DIR!s}"),
#     #         PresetMcpTools.fetch,
#     #     ]:
#     #         rprint(mcp_tool.shell_cmd)
#     #         # MCPClient instantiation is now handled by Agent.init from YAML
#     #         # mcp_client = MCPClient(**mcp_tool.to_common_params())
#     #         # enabled_mcp_clients.append(mcp_client)

#     #     agent = Agent(
#     #         model=DEFAULT_MODEL_NAME,
#     #         # mcp_clients are now internally managed and loaded from YAML
#     #     )
#     #     # await agent.init() # init is now called within invoke

#     #     # This invoke signature has changed.
#     #     # resp = await agent.invoke(
#     #     #     f"爬取 https://news.ycombinator.com 的内容, 并且总结后保存在 {PROJECT_ROOT_DIR / 'output' / 'step3-agent-with-mcp'!s} 目录下的news.md文件中"
#     #     # )
#     #     # rprint(resp)
#     #     rprint("Old example() is commented out. New example using YAML is needed.")
async def example() -> None:
    """
    Example of using the Agent with a YAML configuration file.
    The YAML file is created temporarily for this demonstration.
    """
    yaml_template = f"""
description: "Example prompt for Agent demonstration."

system_prompt: "You are a helpful agent that can access local files and fetch web content."

context: |
  The user is trying to test the agent's capabilities.
  The project root directory is available for file operations.

prompt_text: |
  Please list the files in the directory '{{PROJECT_ROOT_DIR_PLACEHOLDER}}/src/augmented'.
  Then, fetch the content of 'https://jsonplaceholder.typicode.com/todos/1' and tell me the title of the todo.

mcp_servers:
  - id: "local_filesystem"
    type: "local"
    preset_ref: "filesystem"
    # The filesystem preset in McpToolInfo by default operates relative to where the MCP server is run.
    # PresetMcpTools.filesystem already appends PROJECT_ROOT_DIR if it's configured to do so,
    # making it context-aware for the project structure.
    # If preset_mcp_params_append is needed, it can be added here. For example:
    # preset_mcp_params_append: "--root {PROJECT_ROOT_DIR_PLACEHOLDER}" 
    # However, PresetMcpTools.filesystem is often set up with PROJECT_ROOT_DIR already.

  - id: "local_fetch"
    type: "local"
    preset_ref: "fetch"

  # Example of a placeholder for a true remote MCP server:
  # - id: "actual_remote_service"
  #   type: "remote"
  #   url: "http://your-remote-mcp-server.com/mcp" # Replace with a real URL if testing remote
  #   token_env: "MY_REMOTE_MCP_TOKEN" # Ensure this env var is set if uncommented
"""
    final_yaml_content = yaml_template.replace("{{PROJECT_ROOT_DIR_PLACEHOLDER}}", str(PROJECT_ROOT_DIR))
    
    temp_yaml_filename = "_agent_example_prompt.yaml"
    # For simplicity, creating in PROJECT_ROOT_DIR. tempfile module is an alternative for system temp dir.
    yaml_file_path = PROJECT_ROOT_DIR / temp_yaml_filename

    agent = None  # Initialize agent to None for cleanup purposes
    try:
        with open(yaml_file_path, 'w') as f:
            f.write(final_yaml_content)
        PRETTY_LOGGER.info(f"Temporary YAML config written to: {yaml_file_path}")

        agent = Agent(model=DEFAULT_MODEL_NAME)
        
        rprint(f"\nInvoking agent with YAML: {yaml_file_path}...\n")
        response = await agent.invoke(str(yaml_file_path)) # Ensure path is string
        
        PRETTY_LOGGER.title("AGENT FINAL RESPONSE")
        rprint(response)

    except Exception as e:
        rprint(f"Error during agent example execution: {e!s}")
        # RICH_CONSOLE.print_exception() # For more detailed traceback
        raise  # Re-raise the exception after printing
    finally:
        if agent:
            PRETTY_LOGGER.info("Cleaning up agent...")
            await agent.cleanup()
        
        if os.path.exists(yaml_file_path):
            PRETTY_LOGGER.info(f"Cleaning up temporary YAML file: {yaml_file_path}")
            try:
                os.remove(yaml_file_path)
            except Exception as e:
                rprint(f"Error deleting temporary YAML file {yaml_file_path}: {e!s}")


if __name__ == "__main__":
    asyncio.run(example())
