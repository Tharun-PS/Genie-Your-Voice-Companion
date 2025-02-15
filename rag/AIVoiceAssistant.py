from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb
import os
from dotenv import load_dotenv
import warnings


import warnings
warnings.filterwarnings("ignore")
load_dotenv()

class AIVoiceAssistant:
    def __init__(self):
        self._client = chromadb.PersistentClient(path="../chroma_db")
        self._llm = Gemini(model="models/gemini-pro", request_timeout=120.0, api_key=os.environ["GOOGLE_API_KEY"])
        self._embedding_model = GeminiEmbedding(model="models/gemini-pro", api_key=os.environ["GOOGLE_API_KEY"])
        self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model=self._embedding_model)
        self._index = None
        self._collection = self._client.get_or_create_collection("customer_db")
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"C:\Users\SATHIT\Tharun\voice_assistant_flask\rag\training data.txt"]
            )
            documents = reader.load_data()
            vector_store = ChromaVectorStore(chroma_collection=self._collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
        
    
    def interact_with_llm(self, customer_query):
        # Existing LLM interaction logic
        AgentChatResponse = self._chat_engine.chat(customer_query)
        answer = AgentChatResponse.response
        return answer

    @property
    def _prompt(self):
        return """
         Your name is Buddy.
            If customer asks for date and time, give it.
            If they ask to play music, go to YouTube and play that song.
            You are a professional AI Assistant receptionist working in Service Desk of a Laundry Company.
            Ask questions mentioned inside square brackets one by one to keep the conversation engaging:
            
            Ask Name and Type of Appliance, what issue they are facing and Recommend remedy actions that can be done by Customer End without Technician

            If customer is not able to perform the prompted actions, then book a technician visit and end the conversation by greeting.
            For booking a technician visit, ask the customer for date and time and then book the visit.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            Provide concise and short answers (not more than 10 words), and don't chat with yourself!
            After booking a technician visit, print the schedule visit details by saying [Here are the details of the schedule_visit].
            If the customer says thank you, end the conversation by greeting.
            """    