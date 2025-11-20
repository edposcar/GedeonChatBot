import os
from packaging import version
import openai
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager

# Configurazione e validazione
required_version = version.parse("1.1.1")
current_version = version.parse(openai.__version__)

if current_version < required_version:
    raise ValueError(
        f"Error: OpenAI version {openai.__version__} "
        f"is less than the required version 1.1.1"
    )

# Variabili globali per client e assistant_id
client = None
assistant_id = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestisce l'inizializzazione e la chiusura dell'applicazione"""
    global client, assistant_id
    
    # Startup
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    assistant_id = os.environ.get('ASSISTANT_ID')
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if not assistant_id:
        raise ValueError("ASSISTANT_ID environment variable is not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized successfully")
    
    yield
    
    # Shutdown (se necessario)
    print("Application shutting down")

# Inizializza FastAPI con lifespan
app = FastAPI(lifespan=lifespan)

# Modelli Pydantic
class ChatRequest(BaseModel):
    thread_id: str
    message: str

class StartResponse(BaseModel):
    thread_id: str

class ChatResponse(BaseModel):
    response: str

# Costanti
MAX_POLL_ATTEMPTS = 60  # Timeout dopo 60 secondi
POLL_INTERVAL = 1  # Intervallo di polling in secondi

# Endpoint per iniziare una conversazione
@app.get('/start', response_model=StartResponse)
async def start_conversation():
    """Crea un nuovo thread di conversazione"""
    try:
        thread = client.beta.threads.create()
        print(f"New thread created with ID: {thread.id}")
        return {"thread_id": thread.id}
    except Exception as e:
        print(f"Error creating thread: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create thread: {str(e)}")

# Endpoint per gestire i messaggi
@app.post('/chat', response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """Gestisce un messaggio di chat e restituisce la risposta dell'assistente"""
    thread_id = chat_request.thread_id
    user_input = chat_request.message

    if not thread_id:
        raise HTTPException(status_code=400, detail="Missing thread_id")

    print(f"Processing message for thread {thread_id}: {user_input}")

    try:
        # Aggiungi messaggio dell'utente
        await asyncio.to_thread(
            client.beta.threads.messages.create,
            thread_id=thread_id,
            role="user",
            content=user_input
        )

        # Crea la run
        run = await asyncio.to_thread(
            client.beta.threads.runs.create,
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        # Polling con timeout
        attempts = 0
        while attempts < MAX_POLL_ATTEMPTS:
            run_status = await asyncio.to_thread(
                client.beta.threads.runs.retrieve,
                thread_id=thread_id,
                run_id=run.id
            )
            
            status = run_status.status
            print(f"Run status: {status} (attempt {attempts + 1}/{MAX_POLL_ATTEMPTS})")

            if status == 'completed':
                break
            elif status in ['cancelling', 'cancelled', 'expired', 'failed']:
                error_msg = f"Run ended with status: {status}"
                if status == 'failed' and run_status.last_error:
                    error_msg += f" - Error: {run_status.last_error}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            elif status == 'requires_action':
                raise HTTPException(
                    status_code=500,
                    detail="Run requires action - function calling not implemented"
                )

            await asyncio.sleep(POLL_INTERVAL)
            attempts += 1

        if attempts >= MAX_POLL_ATTEMPTS:
            raise HTTPException(
                status_code=504,
                detail="Request timeout: Assistant did not respond in time"
            )

        # Recupera i messaggi
        messages = await asyncio.to_thread(
            client.beta.threads.messages.list,
            thread_id=thread_id,
            limit=1
        )

        if not messages.data:
            raise HTTPException(status_code=500, detail="No response from assistant")

        # Estrai la risposta
        response_text = messages.data[0].content[0].text.value
        print(f"Assistant response: {response_text[:100]}...")

        return {"response": response_text}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint
@app.get('/health')
async def health_check():
    """Verifica lo stato dell'applicazione"""
    return {
        "status": "healthy",
        "openai_configured": client is not None,
        "assistant_configured": assistant_id is not None
    }