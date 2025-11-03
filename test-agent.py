#!/usr/bin/env python3

from llama_index.core.agent.workflow import FunctionAgent  # type: ignore
from llama_index.llms.ollama import Ollama  # type: ignore
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core import (  # type: ignore
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore  # type: ignore
from llama_index.embeddings.ollama import OllamaEmbedding  # type: ignore

import faiss  # type: ignore

# from IPython.display import Markdown, display
import asyncio
import os
import re
import sys

d = 768
embed_model = "nomic-embed-text"
faiss_index = faiss.IndexFlatL2(d)
llm_model = "llama3.1-8b-long-ctx:latest"
llm_model = "qwen3-long-ctx:latest"
llm_model = "gpt-oss-20b-long-ctx:latest"

llm = Ollama(
    model=llm_model,
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=16000,
)
# Create vector store
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

ollama_embedding = OllamaEmbedding(
    model_name=embed_model,  # Replace with your desired Ollama embedding model
    base_url="http://localhost:11434/",  # Default Ollama server address
)
Settings.embed_model = ollama_embedding
# Check if storage directory exists and load existing index
if os.path.exists("./storage") and os.path.isdir("./storage"):
    print("Loading existing index from storage...")
    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)
else:
    print("Creating new index...")
    # Create a RAG tool using LlamaIndex with metadata
    documents = SimpleDirectoryReader("data", filename_as_id=True).load_data()

    # Add metadata to documents (file path will be automatically included)
    for doc in documents:
        # The file_path is already included as metadata by LlamaIndex
        # You can add additional metadata here if needed
        doc.metadata['source_type'] = 'file'
        doc.metadata['indexed_at'] = str(
            os.path.getctime(doc.metadata.get('file_path', '')))

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    # save index to disk
    print("Saving index to storage...")
    index.storage_context.persist()

# mcp server support
# local_client = BasicMCPClient("/home/serg/.local/bin/memory-mcp-server-go",
#                               args=[])  # stdio
# local_client = BasicMCPClient("basic-memory", args=["mcp"])  # stdio
# local_client = BasicMCPClient("http://localhost:8765/mcp")  # http

# Searxng-mpc git@github.com:ihor-sokoliuk/mcp-searxng.git
local_client = BasicMCPClient("http://localhost:3003/mcp")  # http
mcp_tools = McpToolSpec(client=local_client)


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


def summate(a: float, b: float) -> float:
    """Useful for summing two numbers."""
    return a + b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about HPC news.
    Returns the best matching result with its source."""

    # Get retriever to access individual nodes
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = await retriever.aretrieve(query)

    if not nodes:
        return "No relevant documents found."

    # Prepare candidates for LLM evaluation
    candidates = []
    for i, node in enumerate(nodes):
        filename = "unknown"
        if hasattr(node, 'metadata') and 'file_path' in node.metadata:
            filename = node.metadata['file_path'].split('/')[-1]

        candidates.append({
            'index': i,
            'text': node.text[:500] + "..." if len(node.text) > 500 else node.text,  # noqa: E501
            'filename': filename,
            'score': node.score if hasattr(node, 'score') else 0
        })

    # Create evaluation prompt
    eval_prompt = f"""Given the query: "{query}"

Please evaluate these {len(candidates)} document excerpts and
choose the BEST one that most directly answers the query:

"""

    for i, candidate in enumerate(candidates):
        eval_prompt += f"Option {i+1} (from " \
                       f"{candidate['filename']}):\n{candidate['text']}\n\n"

    eval_prompt += f"""
    Respond with ONLY the number (1-{len(candidates)}) of the
    best option that most directly answers the query."""

    # Use LLM to choose best result
    eval_response = await llm.acomplete(eval_prompt)

    try:
        best_index = int(eval_response.text.strip()) - 1
        if 0 <= best_index < len(candidates):
            best_candidate = candidates[best_index]

            # Generate final answer using the best node
            answer_prompt = f"""
            Based on this document excerpt, answer the query: "{query}"

Document excerpt:
{nodes[best_index].text}

Provide a clear, concise answer based on the information in this document."""

            answer_response = await llm.acomplete(answer_prompt)

            return f"{answer_response.text}\n\n" \
                   f"SOURCE FILES: {best_candidate['filename']}\n\n" \
                   "IMPORTANT: Always mention this source file when" \
                   "responding to the user."
        else:
            # Fallback to first result
            best_candidate = candidates[0]
            answer_prompt = f"""
            Based on this document excerpt, answer the query: "{query}"

Document excerpt:
{nodes[0].text}

Provide a clear, concise answer based on the information in this document."""

            answer_response = await llm.acomplete(answer_prompt)
            return f"{answer_response.text}\n\n" \
                   f"SOURCE FILES: {best_candidate['filename']}\n\n" \
                   "IMPORTANT: Always mention this source file when" \
                   " responding to the user."

    except (ValueError, IndexError):
        # Fallback to first result if evaluation fails
        best_candidate = candidates[0]
        answer_prompt = f"""
        Based on this document excerpt, answer the query: "{query}"

Document excerpt:
{nodes[0].text}

Provide a clear, concise answer based on the information in this document."""

        answer_response = await llm.acomplete(answer_prompt)
        return f"{answer_response.text}\n\n" \
               f"SOURCE FILES: {best_candidate['filename']}\n\n" \
               "IMPORTANT: Always mention this source file" \
               " when responding to the user."


def clean(s: str):
    return re.sub(
        r'<think>.*?</think>\s*', '', s, flags=re.DOTALL)


# Now we can ask questions about the documents or do calculations
async def main():
    # Test simple math first
    # print("Testing simple math...")
    # response = await agent.run("What is 5 + 3 * 11?")
    # cleaned_response = clean(str(response))
    # print("Math response:", cleaned_response)
    # print("\n" + "="*50 + "\n")

    # RAG part
    m_tools = await mcp_tools.to_tool_list_async()
    mem_info = """
You have access to a internet search tool, which has method
`searxng_web_search`, allowing you to search information in the net.

You have access to latest news about HPC, if the question
is related to HPC, IT, computer science, then search documets first.

You may combine all tools you have to get better result.

Before answering questions that might have updated info,
search for the news and context in the net, to give more
comprehencive answer.
    """

    # Create an enhanced workflow with all tools
    agent = FunctionAgent(
        tools=[search_documents]+m_tools,  # summate, multiply
        llm=llm,
        system_prompt=f"""
You are a helpful assistant that can search
through documents to answer questions and search information
in the net. You are the real expert and make your best every
time you work an the anwer.

{mem_info}
IMPORTANT: When you use the search_documents function and it
provides source files, you MUST always include those source file
names in your final response to the user. If you didn't use this
tool, use 'None' as source.
IMPORTANT: always specify the tool or tools list you used for
the answer, if you didn't use a tool, specify 'None'
Format the output as
"Answer: answer text
Sources: filename1.txt, filename2.txt
Tools: tools"
        """,
    )

    # Get question from CLI arguments or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What are the latest news about NVIDIA?"

    print(f"Question: {question}")
    print("="*50)

    response = await agent.run(question)
    # Clean up any <think> tags from the response
    cleaned_response = clean(str(response))
    print("Response:\n", cleaned_response)
    print("\n" + "="*50 + "\n")

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
