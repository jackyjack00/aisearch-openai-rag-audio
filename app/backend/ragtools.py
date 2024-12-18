import re
from typing import Any, List

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from rtmt import RTMiddleTier, Tool, ToolResult, ToolResultDirection

from aiohttp import ClientSession
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import os

load_dotenv()

_search_tool_schema = {
    "type": "function",
    "name": "search",
    "description": "Search the knowledge base. The knowledge base is in Italian, translate to and from Italian if " + \
                   "needed. Results are formatted as a source name first in square brackets, followed by the text " + \
                   "content, and a line with '-----' at the end of each result.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

_grounding_tool_schema = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report use of a source from the knowledge base as part of an answer (effectively, cite the source). Sources " + \
                   "appear in square brackets before each knowledge base passage. Always use this tool to cite sources when responding " + \
                   "with information from the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of source names from last statement actually used, do not include the ones not used to formulate a response"
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}

async def call_orchestrator_auth(query:str, url="http://localhost:9000/post") -> dict:
    """Call the orchestrator with its API standard and authentication.

    Args:
        query (str): textual query
        url (str, optional): URL endpoint to orchestrator. Defaults to "http://localhost:9000/post".

    Returns:
        dict: json response by orchetrator
    """
    async with ClientSession() as session:
        json_to_orchestrator = {
          "execEndpoint": "/search",
          "targetExecutor": "",
          "data": [{"text": query}],
          "parameters": {
             "project_name": "sisbot2.0",
             "id_field": "_id",
             "title_field": "title",
             "content_field": "content",
             "tensor_field": "question.tensor",
             "display_fl": ["context", "metadata.mapping"],
             "modality": "kw|tensor",
             "mm": "40%",
             "top_k": 5,
             "rerankDocs": 100,
             "rerankOperator": "multiply",
             "start": 0,
             "rows": 10,
             "search_fl": ["question.text", "metadata.mapping", "context"]
          }
        }

        headers = {"x-api-key": os.getenv("ORCHESTRATOR_API_KEY")}

        async with session.post(url=url, json=json_to_orchestrator, headers=headers) as response:
            json_response = await response.json()
            return json_response

async def call_orchestrator_stellantis(query:str, url="http://localhost:9000/post") -> dict:
    """Call the orchestrator with its API standard and authentication.

    Args:
        query (str): textual query
        url (str, optional): URL endpoint to orchestrator. Defaults to "http://localhost:9000/post".

    Returns:
        dict: json response by orchetrator
    """
    async with ClientSession() as session:
        json_to_orchestrator = {
          "execEndpoint": "/search",
          "targetExecutor": "",
          "data": [{"text": query}],
          "parameters": {
             "project_name": "stellantiskb",
             "id_field": "globalId",
             "title_field": "title",
             "content_field": "content",
             "tensor_field": "tensor",
             "display_fl": ["content", "question"],
             "modality": "kw|tensor",
             "mm": "40%",
             "top_k": 5,
             "rerankDocs": 100,
             "rerankOperator": "multiply",
             "start": 0,
             "rows": 10,
             "search_fl": ["question", "title", "content"]
          }
        }

        async with session.post(url=url, json=json_to_orchestrator) as response:
            json_response = await response.json()
            return json_response

def clean_html(html_string):
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_string, 'html.parser')
    
    # Remove the <style> and <script> tags
    for script_or_style in soup(['style', 'script', 'meta', 'head', 'title']):
        script_or_style.decompose()
    
    # Extract the body text, removing extra whitespace and non-human-readable characters
    text = soup.get_text(separator="\n", strip=True)
    
    # Remove non-breaking spaces or unwanted unicode characters
    cleaned_text = re.sub(r'\xa0', ' ', text)
    
    # Optionally, remove any specific unwanted characters or patterns, if needed
    # cleaned_text = re.sub(r'pattern_to_remove', '', cleaned_text)
    
    return cleaned_text

def from_orchestrator_response_to_retrieved_docs( json_response:dict )->List[dict]:
    return [ f"[{match["tags"]["id"]}]: {clean_html(match["tags"]["content"])}\n-----\n" for match in json_response["data"][0]["matches"] ]

async def _MAIZE_search_tool( args: Any ) -> ToolResult:
    print(f"Searching for '{args['query']}' in the knowledge base.")
    orchestrator_response = await call_orchestrator_stellantis(args['query'])
    result = from_orchestrator_response_to_retrieved_docs(orchestrator_response)
    #result = "Nessun documento rilevante per la query inviata"
    print( result )

    return ToolResult(result, ToolResultDirection.TO_SERVER)

async def _search_tool(
    search_client: SearchClient, 
    semantic_configuration: str,
    identifier_field: str,
    content_field: str,
    embedding_field: str,
    use_vector_query: bool,
    args: Any) -> ToolResult:
    print(f"Searching for '{args['query']}' in the knowledge base.")
    # Hybrid + Reranking query using Azure AI Search
    vector_queries = []
    if use_vector_query:
        vector_queries.append(VectorizableTextQuery(text=args['query'], k_nearest_neighbors=50, fields=embedding_field))
    search_results = await search_client.search(
        search_text=args['query'], 
        query_type="semantic",
        semantic_configuration_name=semantic_configuration,
        top=5,
        vector_queries=vector_queries,
        select=", ".join([identifier_field, content_field])
    )
    result = ""
    async for r in search_results:
        result += f"[{r[identifier_field]}]: {r[content_field]}\n-----\n"
    return ToolResult(result, ToolResultDirection.TO_SERVER)

KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

# TODO: move from sending all chunks used for grounding eagerly to only sending links to 
# the original content in storage, it'll be more efficient overall
async def _report_grounding_tool(search_client: SearchClient, identifier_field: str, title_field: str, content_field: str, args: Any) -> None:
    sources = [s for s in args["sources"] if KEY_PATTERN.match(s)]
    list = " OR ".join(sources)
    print(f"Grounding source: {list}")
    # Use search instead of filter to align with how detailt integrated vectorization indexes
    # are generated, where chunk_id is searchable with a keyword tokenizer, not filterable 
    search_results = await search_client.search(search_text=list, 
                                                search_fields=[identifier_field], 
                                                select=[identifier_field, title_field, content_field], 
                                                top=len(sources), 
                                                query_type="full")
    
    # If your index has a key field that's filterable but not searchable and with the keyword analyzer, you can 
    # use a filter instead (and you can remove the regex check above, just ensure you escape single quotes)
    # search_results = await search_client.search(filter=f"search.in(chunk_id, '{list}')", select=["chunk_id", "title", "chunk"])

    docs = []
    async for r in search_results:
        docs.append({"chunk_id": r[identifier_field], "title": r[title_field], "chunk": r[content_field]})
    return ToolResult({"sources": docs}, ToolResultDirection.TO_CLIENT)

def attach_rag_tools(rtmt: RTMiddleTier,
    credentials: AzureKeyCredential | DefaultAzureCredential,
    search_endpoint: str, search_index: str,
    semantic_configuration: str,
    identifier_field: str,
    content_field: str,
    embedding_field: str,
    title_field: str,
    use_vector_query: bool
    ) -> None:
    # Get Azure client for search engine
    
    #TO_REVERT
    #if not isinstance(credentials, AzureKeyCredential):
    #    credentials.get_token("https://search.azure.com/.default") # warm this up before we start getting requests
    #search_client = SearchClient(search_endpoint, search_index, credentials, user_agent="RTMiddleTier")

    # Add available tools for RealTime middletier
    #TO_REVERT
    #rtmt.tools["search"] = Tool(schema=_search_tool_schema, target=lambda args: _search_tool(search_client, semantic_configuration, identifier_field, content_field, embedding_field, use_vector_query, args))
    #rtmt.tools["report_grounding"] = Tool(schema=_grounding_tool_schema, target=lambda args: _report_grounding_tool(search_client, identifier_field, title_field, content_field, args))

    rtmt.tools["search"] = Tool(schema=_search_tool_schema, target=lambda args: _MAIZE_search_tool(args))
