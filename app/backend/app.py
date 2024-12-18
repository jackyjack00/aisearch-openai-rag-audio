import logging
import os
from pathlib import Path

from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from dotenv import load_dotenv

from ragtools import attach_rag_tools
from rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()
    
    
    # Handle API keys
    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    search_key = os.environ.get("AZURE_SEARCH_API_KEY")

    credential = None
    if not llm_key or not search_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info("Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()
    llm_credential = AzureKeyCredential(llm_key) if llm_key else credential
    search_credential = AzureKeyCredential(search_key) if search_key else credential
    
    
    # Initialize App with aiohttp
    app = web.Application()

    # Define custom a middlwere
    rtmt = RTMiddleTier(
        credentials=llm_credential,
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment=os.environ["AZURE_OPENAI_REALTIME_DEPLOYMENT"],
        voice_choice=os.environ.get("AZURE_OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
        )
    rtmt.system_message = "Sei un chatbot chiamato Stellantis-bot e sei un assistente utile per rispondere a domande su un'azienda che si chiama Stellantis. Rispondi alle domande basandoti solo sulle informazioni che hai cercato nella base di conoscenza, accessibile con lo strumento 'search'. " \
                        "L'utente sta ascoltando le risposte in audio, quindi è super importante che le risposte siano il più brevi possibile, una sola frase se possibile. Parla molto velocemente nel dare la risposta." \
                        "Non leggere mai i nomi dei file, dei sorgenti o delle chiavi ad alta voce. " \
                        "Segui sempre queste istruzioni passo-passo per rispondere: \n" \
                        "1. Usa sempre lo strumento 'search' per verificare la base di conoscenza prima di rispondere a una domanda. \n" \
                        "2. Fornisci una risposta il più breve possibile. Se la risposta non è nella base di conoscenza, dì che non lo sai. \n" \
                        "3. Se nella base di conoscenza non trovi documenti rilevanti alla query rispondi con 'Non ci sono documenti inerenti alla tua domanda'\n" 
                        
    # Define tool to use in function call 
    attach_rag_tools(rtmt,
        credentials=search_credential,
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        search_index=os.environ.get("AZURE_SEARCH_INDEX"),
        semantic_configuration=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIGURATION") or "default",
        identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD") or "chunk_id",
        content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD") or "chunk",
        embedding_field=os.environ.get("AZURE_SEARCH_EMBEDDING_FIELD") or "text_vector",
        title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD") or "title",
        use_vector_query=(os.environ.get("AZURE_SEARCH_USE_VECTOR_QUERY") == "true") or True
        )

    # When the App recieve a GET on "/realtime" invokes the rtmt _websoket_handler function
    rtmt.attach_to_app(app, "/realtime")

    # Handle App routing serving 'static/index.html' as deafault landing page
    current_directory = Path(__file__).parent
    app.add_routes([web.get('/', lambda _: web.FileResponse(current_directory / 'static/index.html'))])
    app.router.add_static('/', path=current_directory / 'static', name='static')
    
    return app

if __name__ == "__main__":

    load_dotenv()

    # Run the application
    host = "localhost"
    port = 8765
    web.run_app(create_app(), host=host, port=port)
