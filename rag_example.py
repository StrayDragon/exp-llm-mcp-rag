import asyncio
from augmented.agent import Agent
from rich import print as rprint

from augmented.embedding_retriever import EembeddingRetriever
from augmented.mcp_client import MCPClient
from augmented.mcp_tools import PresetMcpTools
from augmented.utils import pretty
from augmented.utils.info import DEFAULT_MODEL_NAME, PROJECT_ROOT_DIR
import os # Added for os.remove
from augmented.vector_store import VectorStoreItem


KNOWLEDGE_BASE_DIR = PROJECT_ROOT_DIR / "output" / "step4-rag" / "kownledge"
# KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True) # Moved to main() for clarity

PRETTY_LOGGER = pretty.ALogger("[RAG]")


async def prepare_knowleage_data():
    PRETTY_LOGGER.title("PREPARE_KNOWLEAGE_DATA")
    if list(KNOWLEDGE_BASE_DIR.glob("*.md")):
        rprint(
            "[green]knowledge base already exists, skip prepare_knowleage_data[/green]"
        )
        return

    yaml_template_prepare = f"""
description: "RAG example: Prepare knowledge base from jsonplaceholder users."
prompt_text: "爬取 https://jsonplaceholder.typicode.com/users 的内容, 在 '{{KNOWLEDGE_BASE_DIR_PLACEHOLDER}}' 目录下为每个人创建一个 Markdown 文件, 文件名为用户姓名 (e.g., Leanne_Graham.md), 保存他们的基本信息 (id, name, username, email, address, phone, website, company)."
mcp_servers:
  - id: "local_filesystem"
    type: "local"
    preset_ref: "filesystem"
  - id: "local_fetch"
    type: "local"
    preset_ref: "fetch"
"""
    final_yaml_prepare = yaml_template_prepare.replace(
        "{{KNOWLEDGE_BASE_DIR_PLACEHOLDER}}", str(KNOWLEDGE_BASE_DIR)
    )
    
    temp_yaml_prepare_path = PROJECT_ROOT_DIR / "_rag_prepare_prompt.yaml"
    agent = None
    try:
        with open(temp_yaml_prepare_path, 'w', encoding='utf-8') as f:
            f.write(final_yaml_prepare)
        
        agent = Agent(model=DEFAULT_MODEL_NAME)
        # Agent.init() is called within agent.invoke()
        resp = await agent.invoke(str(temp_yaml_prepare_path))
        rprint(resp)
    finally:
        if agent:
            await agent.cleanup()
        if os.path.exists(temp_yaml_prepare_path):
            try:
                os.remove(temp_yaml_prepare_path)
            except Exception as e:
                rprint(f"Error deleting temporary YAML file {temp_yaml_prepare_path}: {e!s}")


async def retrieve_context(prompt: str):
    er = EembeddingRetriever("BAAI/bge-m3")
    for path in KNOWLEDGE_BASE_DIR.glob("*.md"):
        document = path.read_text()
        await er.embed_documents(document)

    context: list[VectorStoreItem] = await er.retrieve(prompt)
    PRETTY_LOGGER.title("CONTEXT")
    rprint(context)
    return "\n".join([c.document for c in context])


async def rag():
    prompt = f"根据Bret的信息, 创作一个他的故事, 并且把他的故事保存到 {KNOWLEDGE_BASE_DIR.parent / 'story.md'!s} , 要包含他的基本信息和故事"

    context = await retrieve_context(prompt)

    
    output_story_path = KNOWLEDGE_BASE_DIR.parent / 'story.md'

    yaml_template_rag = f"""
description: "RAG example: Create a story based on user Bret's info."
system_prompt: "You are a creative storyteller who writes engaging narratives."
context: |
  {context.replace("'", "''").replace("{", "{{").replace("}", "}}")} # Escape context for f-string and YAML
prompt_text: "根据Bret的信息, 创作一个他的故事, 并且把他的故事保存到 '{{OUTPUT_STORY_PATH_PLACEHOLDER}}' , 要包含他的基本信息和故事."
mcp_servers:
  - id: "local_filesystem"
    type: "local"
    preset_ref: "filesystem"
"""
    # Using .replace for placeholders to avoid issues with f-string trying to interpret context
    final_yaml_rag = yaml_template_rag.replace(
        "{{OUTPUT_STORY_PATH_PLACEHOLDER}}", str(output_story_path)
    )
    # The context placeholder was already handled by the f-string,
    # but if it had complex structures, it would need careful formatting or a proper YAML library.

    temp_yaml_rag_path = PROJECT_ROOT_DIR / "_rag_story_prompt.yaml"
    agent = None
    try:
        with open(temp_yaml_rag_path, 'w', encoding='utf-8') as f:
            f.write(final_yaml_rag)

        agent = Agent(model=DEFAULT_MODEL_NAME)
        # Agent.init() is called within agent.invoke(), context is now part of the YAML
        resp = await agent.invoke(str(temp_yaml_rag_path))
        rprint(resp)
    finally:
        if agent:
            await agent.cleanup()
        if os.path.exists(temp_yaml_rag_path):
            try:
                os.remove(temp_yaml_rag_path)
            except Exception as e:
                rprint(f"Error deleting temporary YAML file {temp_yaml_rag_path}: {e!s}")

async def main():
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists before operations
    await prepare_knowleage_data()
    await rag()


if __name__ == "__main__":
    asyncio.run(main())
