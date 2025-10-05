#!/usr/bin/env python3

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

import faiss

# from IPython.display import Markdown, display
import asyncio
import os

d = 768
embed_model = "nomic-embed-text"
faiss_index = faiss.IndexFlatL2(d)
llm_model = "llama3.1-8b-long-ctx:latest"

# Create vector store
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

ollama_embedding = OllamaEmbedding(
    model_name=embed_model,  # Replace with your desired Ollama embedding model
    base_url="http://localhost:11434",  # Default Ollama server address
)
Settings.embed_model = ollama_embedding
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

# save index do disk
index.storage_context.persist()
# load
# vector_store = FaissVectorStore.from_persist_dir("./storage")
# storage_context = StorageContext.from_defaults(
#     vector_store=vector_store, persist_dir="./storage"
# )
# index = load_index_from_storage(storage_context=storage_context)


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
            'text': node.text[:500] + "..." if len(node.text) > 500 else node.text,
            'filename': filename,
            'score': node.score if hasattr(node, 'score') else 0
        })

    # Create evaluation prompt
    eval_prompt = f"""Given the query: "{query}"

Please evaluate these {len(candidates)} document excerpts and choose the BEST one that most directly answers the query:

"""

    for i, candidate in enumerate(candidates):
        eval_prompt += f"Option {i+1} (from {candidate['filename']}):\n{candidate['text']}\n\n"

    eval_prompt += """Respond with ONLY the number (1-5) of the best option that most directly answers the query."""

    # Use LLM to choose best result
    eval_response = await llm.acomplete(eval_prompt)

    try:
        best_index = int(eval_response.text.strip()) - 1
        if 0 <= best_index < len(candidates):
            best_candidate = candidates[best_index]

            # Generate final answer using the best node
            answer_prompt = f"""Based on this document excerpt, answer the query: "{query}"

Document excerpt:
{nodes[best_index].text}

Provide a clear, concise answer based on the information in this document."""

            answer_response = await llm.acomplete(answer_prompt)

            return f"{answer_response.text}\n\nSOURCE FILES: {best_candidate['filename']}\n\nIMPORTANT: Always mention this source file when responding to the user."
        else:
            # Fallback to first result
            best_candidate = candidates[0]
            answer_prompt = f"""Based on this document excerpt, answer the query: "{query}"

Document excerpt:
{nodes[0].text}

Provide a clear, concise answer based on the information in this document."""

            answer_response = await llm.acomplete(answer_prompt)
            return f"{answer_response.text}\n\nSOURCE FILES: {best_candidate['filename']}\n\nIMPORTANT: Always mention this source file when responding to the user."

    except (ValueError, IndexError):
        # Fallback to first result if evaluation fails
        best_candidate = candidates[0]
        answer_prompt = f"""Based on this document excerpt, answer the query: "{query}"

Document excerpt:
{nodes[0].text}

Provide a clear, concise answer based on the information in this document."""

        answer_response = await llm.acomplete(answer_prompt)
        return f"{answer_response.text}\n\nSOURCE FILES: {best_candidate['filename']}\n\nIMPORTANT: Always mention this source file when responding to the user."


llm = Ollama(
    model=llm_model,
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="compact",
    similarity_top_k=5,
    include_text=True
)
# Create an enhanced workflow with both tools
agent = FunctionAgent(
    tools=[multiply, summate, search_documents],
    llm=llm,
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.

    IMPORTANT: When you use the search_documents function and it
    provides source files, you MUST always include those source file
    names in your final response to the user. If you didn't use this
    tool, use 'None' as source.
    IMPORTANT: always specify the tool you used for the answer, if you
    didn't use a tool, specify 'None'
    Format the output as
    "Answer: answer text
    Sources: filename1.txt, filename2.txt
    Tools: tool"
    """,
)


# Now we can ask questions about the documents or do calculations
async def main():
    # Test simple math first
    print("Testing simple math...")
    response = await agent.run("What is 5 + 3 * 10?")
    print("Math response:", response)
    print("\n" + "="*50 + "\n")

    # Test search function
    print("Testing search...")
    response = await agent.run("What is the latest news about NVIDIA?")
    print("Search response:\n", response)
    print("\n" + "="*50 + "\n")


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
