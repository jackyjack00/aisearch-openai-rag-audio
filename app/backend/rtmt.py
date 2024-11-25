import asyncio
import json
import logging
from enum import Enum
from typing import Any, Callable, Optional

import aiohttp
from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

logger = logging.getLogger("voicerag")

class ToolResultDirection(Enum):
    TO_SERVER = 1
    TO_CLIENT = 2

class ToolResult:
    text: str
    destination: ToolResultDirection

    def __init__(self, text: str, destination: ToolResultDirection):
        self.text = text
        self.destination = destination

    def to_text(self) -> str:
        if self.text is None:
            return ""
        return self.text if type(self.text) == str else json.dumps(self.text)

class Tool:
    """Abstraction of a Tool necessary for a function call"""
    target: Callable[..., ToolResult]
    schema: Any

    def __init__(self, target: Callable[..., ToolResult], schema: Any):
        """
        Args:
            target (Any): lambda function that represent the functionality of the Tool 
                e.g. lambda args: function(..., args)
            schema (Any): json schema for explaining the function to OpenAI API
        """
        self.target = target # lambda args: function(..., args)
        self.schema = schema # 

class RTToolCall:
    tool_call_id: str
    previous_id: str

    def __init__(self, tool_call_id: str, previous_id: str):
        self.tool_call_id = tool_call_id
        self.previous_id = previous_id

class RTMiddleTier:
    endpoint: str
    deployment: str
    key: Optional[str] = None
    
    # Tools are server-side only for now, though the case could be made for client-side tools
    # in addition to server-side tools that are invisible to the client
    tools: dict[str, Tool] = {}

    # Server-enforced configuration, if set, these will override the client's configuration
    # Typically at least the model name and system message will be set by the server
    model: Optional[str] = None
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    disable_audio: Optional[bool] = None
    voice_choice: Optional[str] = None
    api_version: str = "2024-10-01-preview"
    _tools_pending = {}
    _token_provider = None

    def __init__(self, endpoint: str, deployment: str, credentials: AzureKeyCredential | DefaultAzureCredential, voice_choice: Optional[str] = None):
        # Set variables
        self.endpoint = endpoint
        self.deployment = deployment
        self.voice_choice = voice_choice
        if voice_choice is not None:
            logger.info("Realtime voice choice set to %s", voice_choice)
            
        # Handle API keys
        if isinstance(credentials, AzureKeyCredential):
            self.key = credentials.key
        else:
            self._token_provider = get_bearer_token_provider(credentials, "https://cognitiveservices.azure.com/.default")
            self._token_provider() # Warm up during startup so we have a token cached when the first request arrives

    # Handle function calling
    async def _process_message_to_client(self, msg: str, client_ws: web.WebSocketResponse, server_ws: web.WebSocketResponse) -> Optional[str]:
        # Format message incoming from openAI Server as json
        message = json.loads(msg.data)
        updated_message = msg.data
        # Handle messages based on type 
        if message is not None:
            match message["type"]:
                # Hide all informations 'cause reasons 
                case "session.created":
                    session = message["session"]
                    # Hide the instructions, tools and max tokens from clients, if we ever allow client-side 
                    # tools, this will need updating
                    session["instructions"] = ""
                    session["tools"] = []
                    session["voice"] = self.voice_choice
                    session["tool_choice"] = "none"
                    session["max_response_output_tokens"] = None
                    updated_message = json.dumps(message)

                # Do not propagate messages about function_calling
                case "response.output_item.added":
                    if "item" in message and message["item"]["type"] == "function_call":
                        updated_message = None
                
                # Handle function_calls by creating a RTToolCall in self._tools_pending 
                case "conversation.item.created":
                    # OpenAI want to call a function so create a RTToolCall in self._tools_pending 
                    if "item" in message and message["item"]["type"] == "function_call":
                        item = message["item"]
                        if item["call_id"] not in self._tools_pending:
                            # Now exists { "call_id" : RTToolCall }
                            self._tools_pending[item["call_id"]] = RTToolCall(item["call_id"], message["previous_item_id"])
                        updated_message = None
                        
                    # Hide middletier actions by obscuring the fact that openAI acknowledge the function call
                    elif "item" in message and message["item"]["type"] == "function_call_output":
                        updated_message = None
                        
                # Do not propagate messages about function_calling  
                case "response.function_call_arguments.delta":
                    updated_message = None
                
                # Do not propagate messages about function_calling
                case "response.function_call_arguments.done":
                    updated_message = None

                # Wait for complete creation of response item
                case "response.output_item.done":
                    # OpenAI decided to call a function
                    if "item" in message and message["item"]["type"] == "function_call":
                        # Understand which one 
                        item = message["item"]
                        tool_call = self._tools_pending[message["item"]["call_id"]] # RTToolCall
                        tool = self.tools[item["name"]] # Tool
                        # Understand with which arg
                        args = item["arguments"]
                        # Call it
                        result = await tool.target(json.loads(args))
                        # Forward result to the right endpoint
                        await server_ws.send_json({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": item["call_id"],
                                "output": result.to_text() if result.destination == ToolResultDirection.TO_SERVER else ""
                            }
                        })
                        if result.destination == ToolResultDirection.TO_CLIENT:
                            # TODO: this will break clients that don't know about this extra message, rewrite 
                            # this to be a regular text message with a special marker of some sort
                            await client_ws.send_json({
                                "type": "extension.middle_tier_tool_response",
                                "previous_item_id": tool_call.previous_id,
                                "tool_name": item["name"],
                                "tool_result": result.to_text()
                            })
                        updated_message = None

                case "response.done":
                    if len(self._tools_pending) > 0:
                        self._tools_pending.clear() # Any chance tool calls could be interleaved across different outstanding responses?
                        await server_ws.send_json({
                            "type": "response.create"
                        })
                    if "response" in message:
                        replace = False
                        for i, output in enumerate(reversed(message["response"]["output"])):
                            if output["type"] == "function_call":
                                message["response"]["output"].pop(i)
                                replace = True
                        if replace:
                            updated_message = json.dumps(message)                        

        return updated_message
    
    # Intercept session.update messages, overide the variables and set tools
    async def _process_message_to_server(self, msg: str, ws: web.WebSocketResponse) -> Optional[str]:
        # Format messages incoming from application client as json
        message = json.loads(msg.data)
        updated_message = msg.data
        # Handle messages based on type 
        if message is not None:
            match message["type"]:
                # Handle only "session.update" message type
                case "session.update":
                    # Set up session with defined variables
                    session = message["session"]
                    if self.system_message is not None:
                        session["instructions"] = self.system_message
                    if self.temperature is not None:
                        session["temperature"] = self.temperature
                    if self.max_tokens is not None:
                        session["max_response_output_tokens"] = self.max_tokens
                    if self.disable_audio is not None:
                        session["disable_audio"] = self.disable_audio
                    if self.voice_choice is not None:
                        session["voice"] = self.voice_choice
                        
                    # Define tools available by OpenAI model
                    session["tool_choice"] = "auto" if len(self.tools) > 0 else "none"
                    session["tools"] = [tool.schema for tool in self.tools.values()]
                    
                    # Update message and forward it
                    updated_message = json.dumps(message)
        return updated_message

    # Handle all the messages as a middletier router
    async def _forward_messages(self, ws: web.WebSocketResponse):
        # Now the backend becomes a client for the OpenAI server and contacts it via RealtimeAPI
        async with aiohttp.ClientSession(base_url=self.endpoint) as session:
            # Define headers and param for Azure OpenAI API
            params = { "api-version": self.api_version, "deployment": self.deployment}
            headers = {}
            if "x-ms-client-request-id" in ws.headers:
                headers["x-ms-client-request-id"] = ws.headers["x-ms-client-request-id"]
            if self.key is not None:
                headers = { "api-key": self.key }
            else:
                headers = { "Authorization": f"Bearer {self._token_provider()}" } # NOTE: no async version of token provider, maybe refresh token on a timer?
            
            # Connect with "/openai/realtime" via websocket connection
            async with session.ws_connect("/openai/realtime", headers=headers, params=params) as target_ws:
                
                # How rtmt handle messages to send to OpenAI
                async def from_client_to_server():
                    # For each available aiohttp.WSMsgType.TEXT message
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            # Process it accordingly
                            new_msg = await self._process_message_to_server(msg, ws)
                            if new_msg is not None:
                                # Send it to OpenAI Server
                                await target_ws.send_str(new_msg)
                        else:
                            print("Error: unexpected message type:", msg.type)
                    
                    # Means it is gracefully closed by the client then time to close the target_ws
                    if target_ws:
                        print("Closing OpenAI's realtime socket connection.")
                        await target_ws.close()
                
                # How rtmt handle messages recived from OpenAI        
                async def from_server_to_client():
                    # For each available aiohttp.WSMsgType.TEXT message
                    async for msg in target_ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            # Process it accordingly
                            new_msg = await self._process_message_to_client(msg, ws, target_ws)
                            if new_msg is not None:
                                # Send it to application Client
                                await ws.send_str(new_msg)
                        else:
                            print("Error: unexpected message type:", msg.type)

                try:
                    # Perform async routines a.k.a. handle all the messages as a middletier router
                    await asyncio.gather(from_client_to_server(), from_server_to_client())
                except ConnectionResetError:
                    # Ignore the errors resulting from the client disconnecting the socket
                    pass
            
    # Websocket connection
    async def _websocket_handler(self, request: web.Request):
        # Websocket Handshake when Websocket Request is received on "/realtime"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        # Websocket behaviour
        await self._forward_messages(ws)
        return ws
    
    # When the App recieve a GET on "/realtime" invokes the rtmt _websoket_handler function
    def attach_to_app(self, app, path):
        app.router.add_get(path, self._websocket_handler)
