from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import asyncio
import time
import base64
import os
import numpy as np
import tempfile
from typing import Optional, Dict, List
import uvicorn
import threading
import subprocess
from pydub import AudioSegment
import io

# ENHANCED FASTER IMPORTS
# Suppress warnings and reduce verbose output
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# OLLAMA INTEGRATION - Replace DialoGPT imports
import requests  # For Ollama API calls
from sentence_transformers import SentenceTransformer
import hnswlib
import edge_tts
import chromadb
import librosa
import soundfile as sf
import vosk

print("🚀 BigShip Voice Assistant - OLLAMA Edition")
print("💰 100% Free & Open Source with Hindi Support + Ollama LLM")

# Initialize FastAPI
app = FastAPI(
    title="BigShip Voice Assistant - OLLAMA Ultra-Fast",
    description="Enhanced with Ollama LLM for better responses!",
    version="5.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Configuration (using proven Vosk for accurate STT 
VOSK_MODEL_PATH = r"C:\Users\Asus\Downloads\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"

# OLLAMA CONFIGURATION
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"  # Updated to match actual Ollama model name
OLLAMA_TIMEOUT = 5.0  # Timeout for Ollama requests

ALBERT_MODEL = "paraphrase-albert-small-v2"

INDIAN_VOICES = {
    "priya": "en-IN-NeerjaNeural",
    "ayush": "en-IN-AaravNeural",  
    "kavya": "en-IN-KavyaNeural",
    "arjun": "en-IN-PrabhatNeural",
}

# Hindi voices for bilingual support
HINDI_VOICES = {
    "priya_hindi": "hi-IN-SwaraNeural",     # Female Hindi voice
    "arjun_hindi": "hi-IN-MadhurNeural",    # Male Hindi voice
}

DEFAULT_VOICE = "priya"

# Caches
response_cache = {}
audio_cache = {}
intent_cache = {}
embedding_cache = {}
CACHE_TTL = 7200

# Global state
conversation_state = {
    "is_listening": False,
    "is_processing": False,
    "is_speaking": False,
    "current_websocket": None,
    "conversation_context": [],
    "user_language_preference": "hinglish"
}

# Global models
vosk_model = None
albert_model = None
knowledge_collection = None
hnswlib_index = None
knowledge_docs = []
knowledge_metadata = []

# ===================== OLLAMA FUNCTIONS =====================

import signal
import sys

# Add timeout handling for Ollama
def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        # Reduce timeout to 2 seconds
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            print(f"✅ Ollama connected. Available models: {model_names}")
            
            if OLLAMA_MODEL not in model_names:
                print(f"⚠️ Model '{OLLAMA_MODEL}' not found. Using fallback mode.")
                return None
            return OLLAMA_MODEL
        else:
            print(f"❌ Ollama not responding properly: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("❌ Ollama connection timeout - running in fallback mode")
        return None
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return None
    
# Add graceful shutdown handler
def handle_shutdown(signal, frame):
    print("\n🛑 Shutting down gracefully...")
    if conversation_state["current_websocket"]:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(conversation_state["current_websocket"].close())
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)

def call_ollama_api(prompt, model=None, max_tokens=100):
    """Call Ollama API with timeout and error handling"""
    if model is None:
        model = OLLAMA_MODEL
        
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stop": ["\n\n", "User:", "Human:"]
            }
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            print(f"❌ Ollama API error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("⚠️ Ollama request timed out")
        return None
    except Exception as e:
        print(f"❌ Ollama API call failed: {e}")
        return None

# ===================== ENHANCED INITIALIZATION =====================

def initialize_vosk():
    """Initialize Vosk STT (proven accuracy)"""
    try:
        print(f"📂 Loading Vosk model from: {VOSK_MODEL_PATH}")
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"❌ Vosk model not found at {VOSK_MODEL_PATH}")
            return None

        vosk.SetLogLevel(-1)  # Silent
        model = vosk.Model(VOSK_MODEL_PATH)
        return model
    except Exception as e:
        print(f"❌ Vosk failed: {e}")
        return None

def initialize_ollama():
    """Initialize Ollama connection"""
    try:
        available_model = check_ollama_connection()
        if available_model:
            print(f"✅ Ollama initialized with model: {available_model}")
            return available_model
        else:
            print("❌ Ollama initialization failed")
            return None
    except Exception as e:
        print(f"❌ Ollama initialization error: {e}")
        return None

def initialize_albert_embeddings():
    """Initialize ALBERT embeddings (2.5x faster than MiniLM)"""
    try:
        model = SentenceTransformer(ALBERT_MODEL)
        return model
    except Exception as e:
        print(f"❌ ALBERT failed: {e}")
        return None

def initialize_enhanced_knowledge():
    """
    FIXED: Initialize BigShip knowledge base with PROPER BigShip content
    """
    # Add this line at the beginning of the function
    EXCEL_FILE_PATH = r"C:\Users\Asus\Downloads\Merged_File_with_Sheets.xlsx"
    
    try:
        # Connect to existing ChromaDB folder
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            # Delete old collection if exists
            try:
                chroma_client.delete_collection(name="bigship_knowledge")
            except:
                pass
                
            # Create fresh collection with PROPER BigShip knowledge
            collection = chroma_client.create_collection(name="bigship_knowledge")
            
            # PROPER BigShip knowledge base (this was the main issue)
            bigship_docs = [
                "BigShip is India's leading shipping and logistics platform that connects businesses with reliable courier partners across the country. We provide end-to-end shipping solutions with real-time tracking and competitive rates.",
                
                "BigShip works as a comprehensive platform connecting shippers with courier partners. You can book shipments, compare rates from multiple vendors, track packages in real-time, and manage all your logistics through our unified dashboard and API.",
                
                "BigShip offers extensive services including domestic courier delivery, express shipping, bulk shipments, real-time package tracking, vendor partnership programs, competitive rate calculations, warehousing solutions, and logistics management across India.",
                
                "To become a BigShip vendor or courier partner, register through our platform. We provide complete training, technology integration, customer access, and business growth opportunities for delivery companies looking to expand their reach.",
                
                "BigShip provides real-time shipment tracking with instant SMS and email notifications. Track your packages through our website, mobile app, or integrate tracking into your system using our REST API with live status updates.",
                
                "BigShip offers transparent and competitive shipping rates with no hidden fees. Compare prices from multiple verified courier partners and choose the best option based on delivery time, cost, and service quality for your specific needs.",
                
                "BigShip's logistics solutions include end-to-end supply chain management, inventory tracking, warehouse integration, last-mile delivery optimization, bulk shipping discounts, and comprehensive analytics for businesses of all sizes.",
                
                "BigShip supports both domestic and international shipping with partnerships across major Indian cities and towns. We handle everything from small packages to large bulk shipments with reliable tracking and customer support."
            ]
            
            # Add documents to collection
            for i, doc in enumerate(bigship_docs):
                collection.add(
                    documents=[doc],
                    ids=[f"bigship_doc_{i}"],
                    metadatas=[{"source": "bigship_knowledge", "topic": "shipping_logistics"}]
                )
                
        except Exception as e:
            print(f"⚠️ Collection creation error: {e}")
            collection = None
        
        # Fetch all documents & metadata
        if collection:
            results = collection.get()
            documents = results['documents']
            metadatas = results['metadatas']

            if albert_model and len(documents) > 0:
                # Embed all documents
                embeddings = albert_model.encode(documents)
                dimension = embeddings.shape[1]

                # Create HNSWlib index
                hnsw_index = hnswlib.Index(space='cosine', dim=dimension)
                hnsw_index.init_index(max_elements=len(documents), ef_construction=200, M=16)
                hnsw_index.add_items(embeddings, list(range(len(documents))))
                hnsw_index.set_ef(50)
                
                print(f"✅ FIXED: Initialized proper BigShip knowledge with {len(documents)} docs")
                return collection, hnsw_index, documents, metadatas

        return collection, None, [], []

    except Exception as e:
        print(f"❌ Knowledge initialization failed: {e}")
        return None, None, [], []

# Load enhanced models (suppressed verbose output)
vosk_model = initialize_vosk()
ollama_model = initialize_ollama()  # Initialize Ollama instead of DialoGPT
albert_model = initialize_albert_embeddings()
knowledge_collection, hnswlib_index, knowledge_docs, knowledge_metadata = initialize_enhanced_knowledge()
print("✅ Enhanced components initialized (Vosk STT + Ollama LLM for accuracy)")

# ===================== ENHANCED PROCESSING FUNCTIONS =====================

async def enhanced_transcription(audio_data):
    """
    Robust transcription:
    - stable normalization to target_dBFS
    - minimum duration check in milliseconds
    - use PartialResult/FinalResult appropriately
    """
    try:
        if not vosk_model:
            return "", 0

        start_time = time.time()
        # convert webm/opus to PCM with pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        # standardize to 16k mono 16-bit
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        duration_ms = len(audio)  # pydub gives ms
        # Minimum duration guard (300 ms recommended)
        MIN_DURATION_MS = 300
        if duration_ms < MIN_DURATION_MS:
            print(f"⚠️ Audio too short ({duration_ms}ms) for reliable transcription")
            return "", duration_ms

        # Targeted normalization (stable)
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        # Only apply reasonable gain change
        if abs(change_in_dBFS) > 0.1:
            audio = audio.apply_gain(change_in_dBFS)

        # safety boost if extremely quiet
        if audio.dBFS < -30:
            # smaller boost than before
            audio = audio.apply_gain(8)

        pcm_data = audio.raw_data  # bytes
        # Vosk expects bytes (16k, mono, 16-bit)
        rec = vosk.KaldiRecognizer(vosk_model, 16000.0)

        # Feed in one shot (ok because we've already ensured duration)
        accept = rec.AcceptWaveform(pcm_data)
        if accept:
            raw = rec.Result()
        else:
            # Not a final result — try FinalResult to get best available
            raw = rec.FinalResult()

        result = json.loads(raw)
        transcription = result.get("text", "").strip()

        print(f"✅ Raw transcription: '{transcription}' ({duration_ms}ms, dBFS={audio.dBFS:.1f})")
        return transcription, duration_ms

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return "", 0

def fix_common_stt_errors(query):
    """IMPROVED: Better BigShip-specific corrections"""
    corrections = []
    query_lower = query.lower().strip()
    
    # Enhanced BigShip corrections
    bigship_corrections = {
        # Company name variations
        "bigship": ["big ship", "bid ship", "beek ship", "big cheap", "big shift", "beak ship"],
        
        # Service terms
        "vendor": ["winter", "render", "bender", "fender", "window", "vendor partner", "vendor partners"],
        "courier": ["korean", "career", "warrior", "courier company", "courier partners"],  
        "delivery": ["deliver", "deliveries", "delivery partners"],
        "tracking": ["dragging", "cracking", "track my", "find my"],
        "shipping": ["shifting", "chipping", "shopping"],
        "logistics": ["low district", "logic stick"],
        
        # Question patterns
        "what is bigship": ["what is big ship", "what's bigship", "ward is bigship"],
        "how does bigship": ["how does big ship", "how this bigship"],
        "who are partners": ["who are our partners", "who are delivery partners", "who are vendor partners"],
        "track my package": ["track my border", "find my package", "track my water"],
        "customer support": ["customer sport", "custom support"],
    }
    
    corrections.append(query_lower)  # Original
    
    # Apply corrections
    corrected = query_lower
    changed = False
    
    for correct_term, variations in bigship_corrections.items():
        for variation in variations:
            if variation in corrected:
                corrected = corrected.replace(variation, correct_term)
                changed = True
                print(f"🔧 IMPROVED correction: '{variation}' → '{correct_term}'")
    
    if changed:
        corrections.append(corrected)
    
    return corrections

def enhanced_knowledge_search(query):
    """
    IMPROVED: Better knowledge search with query-specific responses
    """
    global knowledge_collection, hnswlib_index, knowledge_docs, albert_model
    
    if not knowledge_collection or not hnswlib_index or not albert_model or not knowledge_docs:
        print("⚠️ Knowledge search components not available")
        return None, 0
    
    try:
        # First try direct query matching for specific answers
        query_lower = query.lower().strip()
        
        # IMPROVED: Direct question-to-answer mapping
        direct_answers = {
            # Vendor/Partner questions
            "vendor partners": "BigShip partners with verified courier companies across India including Blue Dart, DTDC, Delhivery, Ecom Express, and many regional partners to ensure reliable deliveries.",
            "delivery partners": "Our delivery network includes national couriers like Blue Dart, DTDC, Delhivery, India Post, and over 50+ regional partners covering 25,000+ pin codes across India.",
            "courier partners": "BigShip works with trusted courier partners including Blue Dart, DTDC, Delhivery, Ecom Express, India Post, and regional services to provide comprehensive delivery coverage.",
            
            # Service questions
            "customer support": "BigShip provides 24/7 customer support through phone, email, live chat, and WhatsApp. You can reach us at support@bigship.in or call our helpline for immediate assistance.",
            "track my package": "To track your package, visit bigship.in/track, enter your AWB number, or use our mobile app. You'll get real-time updates via SMS and email notifications.",
            "find my package": "You can find your package by entering the tracking number on our website, mobile app, or by calling our support team. We provide real-time location updates.",
            
            # Company questions  
            "what is bigship": "BigShip is India's smartest shipping platform! We help you send packages at the lowest rates by comparing 15+ courier partners in real-time.",
            "how does bigship work": "BigShip works in 3 steps: 1) Enter shipment details and compare rates from multiple couriers, 2) Book and pay online, 3) Schedule pickup and track in real-time until delivery.",
            
            # Pricing questions
            "shipping rates": "BigShip offers up to 40% lower shipping rates by comparing prices from 15+ courier partners. Rates start from ₹25 for documents and ₹35 for small packages within India.",
        }
        
        # Check for direct matches first
        for key, answer in direct_answers.items():
            if key in query_lower:
                print(f"✅ Direct answer match: '{key}' -> confidence: 0.95")
                return answer, 0.95
        
        # If no direct match, use embedding search
        query_emb = albert_model.encode([query])[0]
        labels, distances = hnswlib_index.knn_query(query_emb, k=min(3, len(knowledge_docs)))
        
        idx = labels[0][0]
        distance = distances[0][0]
        confidence = max(0, 1 - distance)
        
        answer = knowledge_docs[idx] if idx < len(knowledge_docs) else None
        
        print(f"📖 Embedding search: query='{query}', confidence={confidence:.3f}")
        
        # Try corrections if confidence is low
        if confidence < 0.4:
            corrections = fix_common_stt_errors(query)
            for corrected_query in corrections[1:]:
                # Check direct answers for corrections
                for key, direct_answer in direct_answers.items():
                    if key in corrected_query.lower():
                        print(f"✅ Correction match: '{corrected_query}' -> '{key}'")
                        return direct_answer, 0.85
                
                # Try embedding search with correction
                try:
                    corrected_emb = albert_model.encode([corrected_query])[0]
                    corr_labels, corr_distances = hnswlib_index.knn_query(corrected_emb, k=1)
                    
                    if len(corr_labels[0]) > 0:
                        corr_idx = corr_labels[0][0]
                        corr_confidence = max(0, 1 - corr_distances[0][0])
                        
                        if corr_confidence > confidence:
                            print(f"✅ Better correction match: conf={corr_confidence:.2f}")
                            return knowledge_docs[corr_idx], corr_confidence
                except:
                    continue
        
        return answer, confidence
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return None, 0

def simple_knowledge_search(query):
    """Simple fallback knowledge search"""
    try:
        if not knowledge_collection:
            return None, 0
        
        # Use ChromaDB's built-in search
        results = knowledge_collection.query(
            query_texts=[query],
            n_results=1
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            return results['documents'][0][0], 0.7
        
        return None, 0
        
    except Exception as e:
        print(f"❌ Simple search error: {e}")
        return None, 0

def generate_contextual_response(query):
    """IMPROVED: More specific contextual responses"""
    query_lower = query.lower().strip()
    
    # More specific BigShip responses
    specific_responses = {
        # Vendor/Partner related
        "vendor": "BigShip partners with 15+ verified courier companies including Blue Dart, DTDC, Delhivery, Ecom Express, and India Post. Want to become a partner? Visit our vendor registration page.",
        "partner": "Our delivery partners include national couriers like Blue Dart, DTDC, and 50+ regional services covering 25,000+ pin codes. We ensure reliable delivery across India.",
        "courier": "BigShip works with trusted couriers including Blue Dart, DTDC, Delhivery, Ecom Express for fast, reliable deliveries. Compare rates from all partners instantly.",
        
        # Service related
        "track": "Track your package easily! Visit bigship.in/track, enter your AWB number, or use our mobile app. Get real-time SMS and email updates on delivery status.",
        "support": "Need help? Contact BigShip support 24/7 at support@bigship.in, call our helpline, or use live chat on our website. We're here to assist you!",
        "delivery": "BigShip ensures fast, secure deliveries across India through our network of verified courier partners. Most packages deliver within 2-7 days depending on location.",
        
        # Business related  
        "rate": "Get the best shipping rates! BigShip compares prices from 15+ couriers to save you up to 40%. Rates start from ₹25 for documents, ₹35 for small packages.",
        "price": "BigShip offers competitive pricing with no hidden fees. Compare real-time rates from multiple couriers and choose the best option for your budget and timeline.",
        "cost": "Save money with BigShip! We negotiate bulk rates with couriers and pass savings to you. Get instant quotes and pay only after successful delivery.",
        
        # Service features
        "api": "BigShip offers robust APIs for seamless integration. Automate your shipping with our developer-friendly API documentation and 24/7 technical support.",
        "bulk": "For bulk shipping, BigShip offers special enterprise rates and dedicated account management. Handle large volumes with ease and get priority support.",
    }
    
    # Find most relevant response
    for keyword, response in specific_responses.items():
        if keyword in query_lower:
            return response
    
    # Question type responses
    if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
        if "bigship" in query_lower or "big ship" in query_lower:
            return "BigShip is India's #1 shipping platform! We help you send packages at the lowest rates by comparing 15+ courier partners. Get instant quotes, book online, and track real-time!"
        elif "work" in query_lower:
            return "BigShip works in 3 simple steps: 1) Enter package details and get instant rate comparison, 2) Choose best courier and pay online, 3) Schedule pickup and track until delivery!"
    
    # Greetings with more info
    if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "namaste"]):
        return "Hello! Welcome to BigShip - India's smartest shipping platform! I can help you with rates, tracking, courier partners, bulk shipping, and more. What do you need?"
    
    # Default with more specificity
    return "I'm BigShip's AI assistant! I can help you with shipping rates, package tracking, courier partners, bulk discounts, API integration, and account support. What specific service interests you?"

async def enhanced_llm_response(user_input, context=""):
    """ENHANCED: Ollama-powered LLM response generation"""
    try:
        if not ollama_model:
            print("⚠️ Ollama not available, using contextual response")
            return generate_contextual_response(user_input)
            
        start_time = time.time()
        
        # Validate and clean input
        user_input_clean = user_input.strip()
        if not user_input_clean or len(user_input_clean) < 2:
            return generate_contextual_response(user_input)
        
        # Create BigShip-specific prompt for Ollama
        system_prompt = """You are BigShip's AI assistant. BigShip is India's leading shipping and logistics platform. 

Key information:
- We connect businesses with 15+ courier partners (Blue Dart, DTDC, Delhivery, Ecom Express, etc.)
- We offer up to 40% savings on shipping rates
- We provide real-time tracking, bulk discounts, API integration
- We cover 25,000+ pin codes across India
- Rates start from ₹25 for documents, ₹35 for small packages

Respond helpfully and concisely (max 2 sentences) about BigShip services."""

        prompt = f"{system_prompt}\n\nUser question: {user_input_clean}\n\nBigShip Assistant:"
        
        # Call Ollama API in a thread to avoid blocking
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                call_ollama_api, 
                prompt, 
                ollama_model, 
                80  # Max tokens for concise responses
            )
            
            if response and len(response.strip()) > 5:
                # Clean up the response
                answer = response.strip()
                
                # Remove any system prompt echoing
                if "BigShip Assistant:" in answer:
                    answer = answer.split("BigShip Assistant:")[-1].strip()
                
                # Limit response length for TTS
                if len(answer) > 150:
                    sentences = answer.split('. ')
                    if len(sentences) > 1:
                        answer = '. '.join(sentences[:2]) + '.'
                    else:
                        answer = answer[:147] + "..."
                
                processing_time = time.time() - start_time
                print(f"⚡ Ollama response: {processing_time:.3f}s - '{answer}'")
                
                return answer
            else:
                print("⚠️ Ollama response too short, using contextual")
                return generate_contextual_response(user_input_clean)
                
        except Exception as e:
            print(f"❌ Ollama API error: {e}")
            return generate_contextual_response(user_input_clean)
        
    except Exception as e:
        print(f"❌ LLM processing error: {e}")
        return generate_contextual_response(user_input)

def detect_language(text):
    """Detect if text is in Hindi or English"""
    hindi_keywords = [
        'नमस्ते', 'नमस्कार', 'हैलो', 'कैसे', 'हैं', 'आप', 'क्या', 'है', 'मैं', 'हूँ', 'की', 'का', 'को', 'се', 'में',
        'bigship', 'shipping', 'courier', 'logistics', 'vendor', 'tracking', 'delivery', 'rates'
    ]
    
    # Count Hindi characters (Devanagari script)
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len(text.replace(' ', ''))
    
    # If more than 30% Hindi characters or contains Hindi keywords
    if (hindi_chars / max(total_chars, 1)) > 0.3 or any(keyword in text for keyword in hindi_keywords):
        return 'hindi'
    return 'english'

def generate_fallback_response(user_input):
    """Generate professional multilingual fallback responses"""
    # Use the contextual response generator as fallback
    return generate_contextual_response(user_input)

async def enhanced_text_to_speech(text, voice=DEFAULT_VOICE, language='english'):
    """Enhanced TTS with speed optimization"""
    try:
        start_time = time.time()
        
        # Allow reasonable response length
        if len(text) > 250:
            sentences = text.split('. ')
            text = '. '.join(sentences[:2]) + '.'
            if len(text) > 250:
                text = text[:247] + "..."
        
        # Check cache first
        cache_key = f"{text[:50]}_{voice}_{language}"
        if cache_key in audio_cache:
            cached = audio_cache[cache_key]
            if time.time() - cached['timestamp'] < CACHE_TTL:
                print(f"⚡ TTS cache hit: {time.time() - start_time:.3f}s")
                return cached['audio_base64']
        
        # Choose voice based on language
        if language == 'hindi':
            voice_id = HINDI_VOICES.get("priya_hindi", "hi-IN-SwaraNeural")
        else:
            voice_id = INDIAN_VOICES.get(voice, INDIAN_VOICES[DEFAULT_VOICE])
        
        # Generate TTS
        communicate = edge_tts.Communicate(
            text, 
            voice_id, 
            rate="+10%",
            volume="+5%"
        )
        
        # Generate audio directly to bytes
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        
        if not audio_bytes:
            print("❌ TTS generation failed")
            return ""
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Cache result
        audio_cache[cache_key] = {
            'audio_base64': audio_base64,
            'timestamp': time.time()
        }
        
        # Clean old cache entries
        if len(audio_cache) > 50:
            oldest_key = min(audio_cache.keys(), key=lambda k: audio_cache[k]['timestamp'])
            del audio_cache[oldest_key]
        
        processing_time = time.time() - start_time
        print(f"⚡ Fast TTS: {processing_time:.2f}s ({len(text)} chars)")
        return audio_base64
        
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return ""
# ===================== ENHANCED MAIN PROCESSING =====================

async def enhanced_voice_processing(audio_data):
    """ENHANCED: Voice processing with Ollama integration"""
    try:
        total_start = time.time()
        
        # Update state
        conversation_state["is_processing"] = True
        await broadcast_state_update()
        
        # Step 1: Enhanced transcription
        transcription, stt_time = await enhanced_transcription(audio_data)
        
        if not transcription or len(transcription.strip()) < 2:
            print("⚠️ Transcription failed or too short")
            conversation_state["is_processing"] = False
            await broadcast_state_update()
            return None
        
        print(f"✅ Initial transcription: '{transcription}'")
        
        # Step 2: Apply corrections (FIXED - no duplicates)
        corrections = fix_common_stt_errors(transcription)
        if len(corrections) > 1:
            transcription = corrections[1]  # Use first correction
            print(f"🔧 Using correction: '{transcription}'")
        
        # Step 3: Knowledge search with FIXED threshold
        print(f"🔍 Searching knowledge for: '{transcription}'")
        
        try:
            knowledge_answer, confidence = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, enhanced_knowledge_search, transcription), 
                timeout=2.0
            )
        except:
            knowledge_answer, confidence = None, 0
        
        # FIXED: Lower confidence threshold from 0.65 to 0.35
        if knowledge_answer and confidence > 0.35:
            response_text = knowledge_answer.strip()
            
            # Limit length
            if len(response_text) > 200:
                sentences = response_text.split('. ')
                if len(sentences) > 1:
                    response_text = '. '.join(sentences[:2]) + '.'
                else:
                    response_text = response_text[:197] + "..."
            
            print(f"✅ OLLAMA: Using knowledge base (conf: {confidence:.2f})")
        else:
            # Use Ollama LLM response
            print(f"⚠️ Knowledge base confidence low ({confidence:.2f}). Using Ollama LLM.")
            response_text = await enhanced_llm_response(transcription)
        
        # Step 4: Generate TTS - FIXED parameter order
        try:
            audio_base64 = await asyncio.wait_for(
                enhanced_text_to_speech(response_text, voice=DEFAULT_VOICE, language='english'),
                timeout=5.0
            )
        except:
            audio_base64 = ""
        
        processing_time = time.time() - total_start
        print(f"⚡ OLLAMA Total processing: {processing_time:.2f}s")
        
        # Update conversation context
        conversation_state["conversation_context"].append({
            "user": transcription,
            "assistant": response_text,
            "timestamp": time.time()
        })
        
        # Keep only last 5 exchanges
        if len(conversation_state["conversation_context"]) > 5:
            conversation_state["conversation_context"] = conversation_state["conversation_context"][-5:]
        
        conversation_state["is_processing"] = False
        conversation_state["is_speaking"] = True
        await broadcast_state_update()
        
        return {
            "transcription": transcription,
            "response": response_text,
            "audio_base64": audio_base64,
            "processing_time": processing_time,
            "stt_time": stt_time
        }
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        conversation_state["is_processing"] = False
        await broadcast_state_update()
        return None

async def broadcast_state_update():
    """Broadcast state updates to frontend"""
    if conversation_state["current_websocket"]:
        try:
            await conversation_state["current_websocket"].send_json({
                "type": "state_update",
                "is_listening": conversation_state["is_listening"],
                "is_processing": conversation_state["is_processing"],
                "is_speaking": conversation_state["is_speaking"]
            })
        except:
            pass
# ===================== FRONTEND =====================

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BigShip Voice Assistant</title>
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body { 
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            min-height: 100vh; 
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .voice-orb-container {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100vw;
            height: 100vh;
        }

        .voice-orb {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #60a5fa, #3b82f6, #1d4ed8);
            box-shadow: 
                0 0 60px rgba(59, 130, 246, 0.4),
                0 0 120px rgba(59, 130, 246, 0.2),
                inset 0 0 60px rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .voice-orb::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: conic-gradient(from 0deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: orbRotate 4s linear infinite;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .voice-orb.listening {
            background: radial-gradient(circle at 30% 30%, #34d399, #10b981, #059669);
            box-shadow: 
                0 0 80px rgba(16, 185, 129, 0.6),
                0 0 160px rgba(16, 185, 129, 0.3),
                inset 0 0 60px rgba(255, 255, 255, 0.1);
            animation: listeningPulse 2s ease-in-out infinite;
        }

        .voice-orb.listening::before {
            opacity: 1;
        }

        .voice-orb.processing {
            background: radial-gradient(circle at 30% 30%, #fbbf24, #f59e0b, #d97706);
            box-shadow: 
                0 0 80px rgba(245, 158, 11, 0.6),
                0 0 160px rgba(245, 158, 11, 0.3),
                inset 0 0 60px rgba(255, 255, 255, 0.1);
            animation: processingPulse 1s ease-in-out infinite;
        }

        .voice-orb.speaking {
            background: radial-gradient(circle at 30% 30%, #a78bfa, #8b5cf6, #7c3aed);
            box-shadow: 
                0 0 100px rgba(139, 92, 246, 0.7),
                0 0 200px rgba(139, 92, 246, 0.4),
                inset 0 0 60px rgba(255, 255, 255, 0.1);
            animation: speakingPulse 0.8s ease-in-out infinite;
        }

        .voice-orb.speaking::before {
            opacity: 0.7;
            animation: orbRotate 2s linear infinite;
        }

        /* Audio visualization rings */
        .audio-rings {
            position: absolute;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .audio-ring {
            position: absolute;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            opacity: 0;
        }

        .audio-ring.active {
            animation: ringExpand 2s ease-out infinite;
        }

        @keyframes listeningPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        @keyframes processingPulse {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.05) rotate(180deg); }
        }

        @keyframes speakingPulse {
            0%, 100% { transform: scale(1); }
            25% { transform: scale(1.15); }
            75% { transform: scale(1.05); }
        }

        @keyframes orbRotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes ringExpand {
            0% {
                width: 200px;
                height: 200px;
                opacity: 0.6;
            }
            100% {
                width: 400px;
                height: 400px;
                opacity: 0;
            }
        }

        /* Subtle background particles */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }

        .particle {
            position: absolute;
            width: 2px;
            height: 2px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            animation: particleFloat 8s ease-in-out infinite;
        }

        @keyframes particleFloat {
            0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0; }
            50% { opacity: 1; }
        }

        /* Auto-start pulse animation */
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }

        /* Title */
        .title {
            position: absolute;
            top: 50px;
            left: 50%;
            transform: translateX(-50%);
            color: #ffffff;
            text-align: center;
            z-index: 10;
            opacity: 0.9;
        }

        .title h1 {
            font-size: 2rem;
            font-weight: 300;
            letter-spacing: 2px;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        }

        /* Permission overlay */
        .permission-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }

        .permission-content {
            text-align: center;
            color: white;
            max-width: 400px;
            padding: 40px;
        }

        .permission-title {
            font-size: 24px;
            margin-bottom: 16px;
            font-weight: 600;
        }

        .permission-subtitle {
            font-size: 16px;
            opacity: 0.8;
            margin-bottom: 32px;
            line-height: 1.5;
        }

        .permission-button {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            border: none;
            border-radius: 50px;
            color: white;
            padding: 16px 32px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }

        .permission-button:hover {
            transform: scale(1.05);
        }

        .hidden {
            display: none;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .title h1 {
                font-size: 1.5rem;
            }
            
            .voice-orb {
                width: 150px;
                height: 150px;
            }
        }

    </style>
</head>
<body>
    <div class="particles" id="particles"></div>

    <div class="permission-overlay hidden" id="permissionOverlay">
        <div class="permission-content">
            <div class="permission-title">Enable Voice Interaction</div>
            <div class="permission-subtitle">BigShip AI needs microphone access for voice conversations</div>
            <button class="permission-button" onclick="requestMicrophoneAccess()">Enable Microphone</button>
        </div>
    </div>

    <div class="title">
        <h1>BigShip Voice Assistant</h1>
    </div>

    <div class="voice-orb-container">
        <div class="voice-orb" id="voiceOrb" onclick="handleOrbClick()">
            <div class="audio-rings" id="audioRings"></div>
        </div>
    </div>

    <script>
        let ws = null;
        let mediaRecorder = null;
        let audioStream = null;
        let isListening = false;
        let isProcessing = false;
        let isPlayingAudio = false;
        let hasGreeted = false;
        let audioChunks = [];
        let currentAudio = null;
        let micPermissionGranted = false;
        let silenceDetectionTimer = null;
        let isAudioPlaying = false; // Global flag to prevent overlapping audio
        // Silence detection variables
        let silenceTimer = null;
        let audioContext = null;
        let analyser = null;
        let dataArray = null;
        let source = null;
        let silenceThreshold = 30; // Adjust this value (lower = more sensitive)
        let silenceDelay = 1500; // Wait 1.5 seconds of silence before stopping


        // Initialize on page load
        window.addEventListener('load', function() { 
            console.log('🚀 BigShip Voice Assistant initializing...');
            createParticles();
            enableAudioContext();
            
            // Auto-check microphone permission instead of always showing overlay
            checkAndRequestMicrophoneAccess();
        });

        function createParticles() {
            const particles = document.getElementById('particles');
            for (let i = 0; i < 30; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 8 + 's';
                particle.style.animationDuration = (4 + Math.random() * 8) + 's';
                particles.appendChild(particle);
            }
        }

        function enableAudioContext() {
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                if (audioContext.state === 'suspended') {
                    audioContext.resume();
                }
                console.log('✅ Audio context enabled');
            } catch (e) {
                console.log('⚠️ Audio context setup:', e);
            }
        }

        async function checkAndRequestMicrophoneAccess() {
            try {
                // Check localStorage for user preference
                const userPreference = localStorage.getItem('bigship_mic_preference');
                
                // Check if microphone permission is already granted
                const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
                
                if (permissionStatus.state === 'granted') {
                    console.log('✅ Microphone already permitted - auto-starting');
                    
                    // Show brief loading indicator
                    showAutoStartMessage();
                    
                    await autoRequestMicrophoneAccess();
                } else if (permissionStatus.state === 'prompt') {
                    // Only show overlay if user hasn't indicated they don't want to be asked
                    if (userPreference !== 'auto_deny') {
                        console.log('❓ Microphone permission needed - showing overlay');
                        document.getElementById('permissionOverlay').classList.remove('hidden');
                    } else {
                        console.log('⚠️ User previously declined - manual activation required');
                        showManualActivationMessage();
                    }
                } else {
                    console.log('❌ Microphone permission denied - showing overlay');
                    document.getElementById('permissionOverlay').classList.remove('hidden');
                }
                
                // Listen for permission changes
                permissionStatus.addEventListener('change', function() {
                    if (this.state === 'granted' && !micPermissionGranted) {
                        autoRequestMicrophoneAccess();
                    }
                });
                
            } catch (error) {
                console.log('⚠️ Permission API not supported - showing overlay');
                document.getElementById('permissionOverlay').classList.remove('hidden');
            }
        }

        function showAutoStartMessage() {
            // Clear visual feedback that it's auto-starting
            const orb = document.getElementById('voiceOrb');
            const title = document.querySelector('.title h1');
            
            title.textContent = '🚀 AUTO-STARTING - Setting up microphone...';
            title.style.color = '#3b82f6'; // Blue
            orb.style.animation = 'pulse 1s ease-in-out';
            orb.title = 'Auto-starting voice assistant...';
            
            setTimeout(() => {
                orb.style.animation = '';
                updateOrbState(''); // This will set it to READY state
            }, 2000);
        }

        function showManualActivationMessage() {
            // Show a clear message that manual activation is needed
            const orb = document.getElementById('voiceOrb');
            const title = document.querySelector('.title h1');
            
            title.textContent = '👆 CLICK ORB - Manual activation needed';
            title.style.color = '#ef4444'; // Red
            orb.title = 'Click the orb to enable voice assistant';
            orb.style.opacity = '0.7';
            
            // Restore normal state after a while
            setTimeout(() => {
                orb.style.opacity = '1';
                updateOrbState(''); // Back to ready state
            }, 5000);
        }

        async function autoRequestMicrophoneAccess() {
            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: false,
                        sampleRate: 16000,
                        channelCount: 1
                    } 
                });
                
                console.log('✅ Microphone auto-granted');
                micPermissionGranted = true;
                
                // Remember user's permission preference
                localStorage.setItem('bigship_mic_preference', 'granted');
                
                // Hide permission overlay if it's showing
                document.getElementById('permissionOverlay').classList.add('hidden');
                
                // Setup MediaRecorder and connect
                setupMediaRecorder();
                connectWebSocket();
                
                return true;
                
            } catch (error) {
                console.log('❌ Auto microphone access failed:', error);
                
                // Remember if user explicitly denied
                if (error.name === 'NotAllowedError') {
                    localStorage.setItem('bigship_mic_preference', 'denied');
                }
                
                // Show overlay if auto-access fails
                document.getElementById('permissionOverlay').classList.remove('hidden');
                return false;
            }
        }

        async function requestMicrophoneAccess() {
            console.log('🎤 User requesting microphone permission...');
            
            // Clear any previous denial preference
            localStorage.removeItem('bigship_mic_preference');
            
            const success = await autoRequestMicrophoneAccess();
            
            if (!success) {
                alert('Microphone access is required for voice interaction. Please allow access and reload.');
            }
        }

        function setupMediaRecorder() {
            if (!audioStream) {
                console.warn('⚠️ No audio stream available');
                return;
            }
            
            try {
                mediaRecorder = new MediaRecorder(audioStream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = function() {
                    console.log('🛑 Recording stopped, processing...');
                    processRecordedAudio();
                };
                
                console.log('✅ MediaRecorder setup complete');
                
            } catch (error) {
                console.error('❌ MediaRecorder setup failed:', error);
            }
        }

        function connectWebSocket() {
            try {
                ws = new WebSocket('ws://' + window.location.host + '/ws');
                
                ws.onopen = function() { 
                    console.log('✅ WebSocket connected');
                    
                    // Auto-greet after connection
                    setTimeout(() => {
                        if (!hasGreeted) {
                            autoGreet();
                        }
                    }, 1000);
                };
                
                ws.onmessage = function(event) { 
                    handleWebSocketMessage(JSON.parse(event.data)); 
                };
                
                ws.onerror = function(error) { 
                    console.error('❌ WebSocket error:', error);
                };
                
                ws.onclose = function() { 
                    console.warn('🔌 WebSocket closed - reconnecting...');
                    setTimeout(connectWebSocket, 3000); 
                };
            } catch (error) {
                console.error('❌ WebSocket connection failed:', error);
            }
        }

        function handleWebSocketMessage(data) {
            console.log('📨 Received message:', data.type);
            
            switch(data.type) {
                case 'state_update':
                    if (data.is_listening) {
                        updateOrbState('listening');
                    } else if (data.is_processing) {
                        updateOrbState('processing');
                    } else if (data.is_speaking) {
                        updateOrbState('speaking');
                    } else {
                        updateOrbState('');
                    }
                    break;
                    
                case 'response':
                    handleResponse(data);
                    break;
                    
                case 'error':
                    console.error('❌ Backend error:', data.message);
                    // Reset all states on error
                    isProcessing = false;
                    isListening = false;
                    isPlayingAudio = false;
                    updateOrbState('');
                    
                    // Quick restart after error for natural conversation flow
                    setTimeout(() => {
                        if (!isListening && !isProcessing && !isPlayingAudio && micPermissionGranted) {
                            console.log('🔄 Restarting after error...');
                            startListening();
                        }
                    }, 300);  // Much faster error recovery
                    break;
            }
        }

        function updateOrbState(state) {
            const orb = document.getElementById('voiceOrb');
            const title = document.querySelector('.title h1');
            
            orb.className = 'voice-orb ' + (state || '');
            
            // Clear, actionable status messages
            switch(state) {
                case 'listening':
                    title.textContent = '🎤 SPEAK NOW - I am listening...';
                    title.style.color = '#10b981'; // Green
                    orb.title = 'Speaking... Click to stop';
                    break;
                    
                case 'processing':
                    title.textContent = '🤔 PROCESSING - Please wait...';
                    title.style.color = '#f59e0b'; // Yellow
                    orb.title = 'Processing your request...';
                    break;
                    
                case 'speaking':
                    title.textContent = '🗣️ AI RESPONDING - Listen carefully...';
                    title.style.color = '#8b5cf6'; // Purple
                    orb.title = 'AI is speaking...';
                createAudioRings();
                    break;
                    
                default:
                    title.textContent = '✨ READY - Click orb or press SPACE to talk';
                    title.style.color = '#ffffff'; // White
                    orb.title = 'Ready! Click to start talking or press SPACE';
                clearAudioRings();
                    break;
            }
        }

        function createAudioRings() {
            const audioRings = document.getElementById('audioRings');
            audioRings.innerHTML = '';
            
            for (let i = 0; i < 3; i++) {
                const ring = document.createElement('div');
                ring.className = 'audio-ring active';
                ring.style.animationDelay = (i * 0.6) + 's';
                audioRings.appendChild(ring);
            }
        }

        function clearAudioRings() {
            const audioRings = document.getElementById('audioRings');
            audioRings.innerHTML = '';
        }

        function handleResponse(data) {
            console.log('🎯 Response received:', data);
            
            // Reset processing state immediately
            isProcessing = false;
            
            if (data.audio_base64 && data.transcription && data.response) {
                // Valid response with audio
                playAudio(data.audio_base64);
            } else if (data.transcription && data.response) {
                // Valid response but no audio - wait before restart
                console.log('📝 Text response (no audio):', data.response);
                updateOrbState('');
                setTimeout(() => {
                    if (!isListening && !isProcessing && !isPlayingAudio && !isAudioPlaying && micPermissionGranted) {
                        startListening();
                    }
                }, 500);  // Wait before restart to prevent overlap
            } else {
                // Empty or failed response - wait before restart
                console.log('⚠️ Empty response - resetting and restarting');
                updateOrbState('');
                setTimeout(() => {
                    if (!isListening && !isPlayingAudio && !isProcessing && !isAudioPlaying && micPermissionGranted) {
                        startListening();
                    }
                }, 500);  // Wait before restart to prevent overlap
            }
        }

        function playAudio(audioBase64) {
            console.log('🔊 Playing AI response...');
            
            // CRITICAL: Prevent multiple audio playback
            if (isAudioPlaying) {
                console.log('⚠️ Audio already playing - skipping new audio');
                return;
            }
            
            // CRITICAL: Stop any currently playing audio first
            if (currentAudio) {
                console.log('🛑 Stopping previous audio...');
                currentAudio.pause();
                currentAudio.currentTime = 0;
                currentAudio = null;
            }
            
            // Set global flag to prevent overlap
            isAudioPlaying = true;
            isPlayingAudio = true;
            isProcessing = false;
            
            // Small delay to ensure previous audio is fully stopped
            setTimeout(() => {
                updateOrbState('speaking');
                
                currentAudio = new Audio();
                currentAudio.volume = 1.0;
                currentAudio.src = 'data:audio/wav;base64,' + audioBase64;
                
                currentAudio.onended = function() {
                    console.log('✅ Audio playback finished');
                    currentAudio = null;
                    isAudioPlaying = false;
                    audioPlaybackComplete();
                };
                
                currentAudio.onerror = function(e) {
                    console.error('❌ Audio playback error:', e);
                    currentAudio = null;
                    isAudioPlaying = false;
                    audioPlaybackComplete();
                };
                
                currentAudio.play().catch(e => {
                    console.error('❌ Audio play failed:', e);
                    currentAudio = null;
                    isAudioPlaying = false;
                    audioPlaybackComplete();
                });
            }, 100); // Small delay to prevent overlap
        }

        function audioPlaybackComplete() {
            isPlayingAudio = false;
            isProcessing = false;
            isAudioPlaying = false; // Clear global flag
            
            // Clear "AI RESPONDING" message immediately
            updateOrbState('');
            
            // Brief visual cue that it's ready for next question
            const title = document.querySelector('.title h1');
            title.textContent = '✅ READY FOR NEXT QUESTION';
            title.style.color = '#10b981'; // Green
            
            // Change back to normal ready state after 1 second
            setTimeout(() => {
                updateOrbState(''); // Back to normal ready state
            }, 1000);
            
            // Wait longer before restarting to prevent overlap issues
            if (micPermissionGranted) {
                setTimeout(() => {
                    if (!isPlayingAudio && !isProcessing && !isListening && !isAudioPlaying) {
                        startListening();
                    }
                }, 800);  // Increased delay to prevent overlap
            }
        }

        function handleOrbClick() {
            if (!micPermissionGranted) {
                document.getElementById('permissionOverlay').classList.remove('hidden');
                return;
            }
            
            if (isListening) {
                stopListening();
            } else if (!isProcessing && !isPlayingAudio) {
                startListening();
            }
        }

        function startListening() {
            if (!mediaRecorder) {
                console.log('⚠️ MediaRecorder not available');
                return;
            }
            
            // Enhanced state checking with auto-recovery
            if (isProcessing || isPlayingAudio || isAudioPlaying) {
                console.log('⚠️ Cannot start listening - busy state');
                // Auto-recovery: reset stuck states after 5 seconds
                setTimeout(() => {
                    if (isProcessing && !isPlayingAudio && !isAudioPlaying) {
                        console.log('🔄 Auto-recovering from stuck processing state');
                        isProcessing = false;
                        updateOrbState('');
                        // Try to start listening again
                        if (!isListening && micPermissionGranted) {
                            startListening();
                        }
                    }
                }, 5000);
                return;
            }
            
            try {
                audioChunks = [];
                mediaRecorder.start();
                isListening = true;
                
                updateOrbState('listening');
                console.log('👂 Started listening...');
                
                // Send listening state to backend
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'start_listening' }));
                }
                
                // Countdown timer for user awareness
                let timeLeft = 2;
                const countdownTimer = setInterval(() => {
                    if (!isListening) {
                        clearInterval(countdownTimer);
                        return;
                    }
                    
                    const title = document.querySelector('.title h1');
                    title.textContent = `🎤 SPEAK NOW - ${timeLeft}s remaining...`;
                    timeLeft--;
                    
                    if (timeLeft < 0) {
                        clearInterval(countdownTimer);
                    }
                }, 1000);
                
                // Auto-stop after 2 seconds of silence for natural conversation flow
                silenceDetectionTimer = setTimeout(() => {
                    if (isListening) {
                        console.log('⏰ Auto-stopping due to silence');
                        clearInterval(countdownTimer);
                        stopListening();
                    }
                }, 3500);  // Faster silence detection for instant conversation
                
            } catch (error) {
                console.error('❌ Failed to start listening:', error);
                isListening = false;
                updateOrbState('');
                // Quick retry after error for natural flow
                setTimeout(() => {
                    if (!isListening && !isProcessing && !isPlayingAudio && micPermissionGranted) {
                        startListening();
                    }
                }, 200);  // Much faster recovery for seamless conversation
            }
        }

        function stopListening() {
            if (!isListening || !mediaRecorder) return;
            
            try {
                mediaRecorder.stop();
                isListening = false;
                
                if (silenceDetectionTimer) {
                    clearTimeout(silenceDetectionTimer);
                    silenceDetectionTimer = null;
                }
                
                updateOrbState('processing');
                console.log('🛑 Stopped listening');
                
            } catch (error) {
                console.error('❌ Failed to stop listening:', error);
                // Reset state on error
                isListening = false;
                updateOrbState('');
            }
        }

        async function processRecordedAudio() {
            if (audioChunks.length === 0) {
                console.log('⚠️ No audio to process');
                updateOrbState('');
                // Wait before restart to prevent overlap
                setTimeout(() => {
                    if (!isListening && !isProcessing && !isPlayingAudio && !isAudioPlaying && micPermissionGranted) {
                        startListening();
                    }
                }, 500);  // Wait before restart to prevent overlap
                return;
            }
            
            isProcessing = true;
            
            try {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                console.log('📤 Sending audio...', audioBlob.size, 'bytes');
                
                // Enhanced connection checking
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const base64Audio = reader.result.split(',')[1];
                        ws.send(JSON.stringify({
                            type: 'audio_data',
                            audio: base64Audio
                        }));
                    };
                    
                    reader.onerror = () => {
                        console.error('❌ FileReader error');
                        isProcessing = false;
                        updateOrbState('');
                        // Instant restart for natural flow
                        setTimeout(() => startListening(), 100);  // Ultra-fast recovery
                    };
                    
                    reader.readAsDataURL(audioBlob);
                    
                    // Safety timeout - auto-reset if no response in 8 seconds
                    setTimeout(() => {
                        if (isProcessing) {
                            console.log('⏰ Processing timeout - auto-recovering');
                            isProcessing = false;
                            updateOrbState('');
                            // Auto-restart listening
                            if (!isListening && !isPlayingAudio && micPermissionGranted) {
                                startListening();
                            }
                        }
                    }, 8000);
                    
                } else {
                    console.error('❌ Connection lost');
                    isProcessing = false;
                    updateOrbState('');
                    // Quick reconnect and restart for natural flow
                    connectWebSocket();
                    setTimeout(() => startListening(), 500);  // Faster reconnection
                }
                
            } catch (error) {
                console.error('❌ Audio processing failed:', error);
                isProcessing = false;
                updateOrbState('');
                // Quick restart on error for natural flow
                setTimeout(() => {
                    if (!isListening && !isPlayingAudio && micPermissionGranted) {
                        startListening();
                    }
                }, 100);  // Ultra-fast error recovery
            }
            
            audioChunks = [];
        }

        function autoGreet() {
            if (hasGreeted) return;
            hasGreeted = true;
            
            console.log('👋 Auto-greeting...');
            enableAudioContext();
            
            // Send auto-greeting request to backend (like backup.py)
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ 
                    type: 'auto_greeting', 
                    data: 'system_greeting' 
                }));
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            if (event.code === 'Space' && !event.repeat) {
                event.preventDefault();
                if (!isListening && !isProcessing && !isPlayingAudio && micPermissionGranted) {
                    startListening();
                }
            }
        });

        document.addEventListener('keyup', function(event) {
            if (event.code === 'Space') {
                event.preventDefault();
                if (isListening) {
                    stopListening();
                }
            }
        });

        console.log('🎤 BigShip Voice Assistant Ready');
        console.log('💡 Pure voice interaction - Click orb or press SPACE to talk');
    </script>
</body>
</html>'''

# ===================== WEBSOCKET ENDPOINT =====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conversation_state["current_websocket"] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "start_listening":
                conversation_state["is_listening"] = True
                await broadcast_state_update()
                
            elif data["type"] == "auto_greeting":
                # NATURAL HUMAN-LIKE GREETING
                greeting = "Hi! I'm your BigShip assistant. I can help you with shipping, logistics, vendor partnerships, and anything about BigShip. Just ask me anything - I'm here to help!"
                
                try:
                    audio_base64 = await enhanced_text_to_speech(greeting, "priya", "english")
                    await websocket.send_json({
                        "type": "response", 
                        "transcription": "auto_greeting",
                        "response": greeting,
                        "audio_base64": audio_base64,
                        "processing_time": 0.5
                    })
                except Exception as e:
                    print(f"❌ Auto-greeting error: {e}")
                    await websocket.send_json({
                        "type": "response", 
                        "transcription": "auto_greeting",
                        "response": greeting,
                        "audio_base64": "",
                        "processing_time": 0.5
                    })
                
            elif data["type"] == "audio_data":
                conversation_state["is_listening"] = False
                conversation_state["is_processing"] = True
                await broadcast_state_update()
                
                try:
                    audio_base64 = data["audio"]
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    result = await enhanced_voice_processing(audio_bytes)
                    
                    if result:
                        await websocket.send_json({
                            "type": "response",
                            "transcription": result["transcription"],
                            "response": result["response"],
                            "audio_base64": result["audio_base64"],
                            "processing_time": result["processing_time"]
                        })
                    else:
                        # NATURAL FEEDBACK FOR AUDIO ISSUES
                        feedback_message = "I didn't catch that. Could you please speak a bit louder and try again?"
                        
                        try:
                            feedback_audio = await enhanced_text_to_speech(feedback_message, "priya", "english")
                            await websocket.send_json({
                                "type": "response",
                                "transcription": "audio_quality_issue",
                                "response": feedback_message,
                                "audio_base64": feedback_audio,
                                "processing_time": 0.5
                            })
                        except:
                            await websocket.send_json({
                                "type": "response",
                                "transcription": "audio_quality_issue",
                                "response": feedback_message,
                                "audio_base64": "",
                                "processing_time": 0.1
                            })
                        
                        conversation_state["is_processing"] = False
                        conversation_state["is_listening"] = False
                        conversation_state["is_speaking"] = False
                        await broadcast_state_update()
                        
                except Exception as e:
                    print(f"❌ Audio processing error: {e}")
                    
                    await websocket.send_json({
                        "type": "error",
                        "message": "Something went wrong - let's try again"
                    })
                
                    conversation_state["is_processing"] = False
                    conversation_state["is_listening"] = False
                    conversation_state["is_speaking"] = False
                    await broadcast_state_update()
                    
    except WebSocketDisconnect:
        conversation_state["current_websocket"] = None

# ===================== API ENDPOINTS =====================

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "version": "5.0.0 - ENHANCED",
        "components": {
            "vosk": vosk_model is not None,
            "ollama": ollama_model is not None,  # Fixed reference
            "albert": albert_model is not None,
            "hnswlib": hnswlib_index is not None,
            "knowledge_base": knowledge_collection is not None
        },
        "enhancements": {
            "stt": "Vosk (proven accuracy)",
            "llm": "Ollama integration",
            "embeddings": "2.5x faster (ALBERT)",
            "search": "2.5x faster (HNSWlib)"
        }
    }

if __name__ == "__main__":
    print("\n🚀 BigShip Voice Assistant - ENHANCED Edition")
    print("📱 http://localhost:8001")
    print("🎤 100% Hands-free voice platform")
    print("🌐 Hindi + English support")
    print("⚡ Enhanced with Vosk STT (accurate) + Ollama LLM")
    print("🎯 Just speak after allowing microphone!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)