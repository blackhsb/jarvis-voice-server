"""
Jarvis Real-time Voice Server
WebSocket-based bidirectional audio for Vobiz <Stream>
Low-latency pipeline: Audio -> STT -> LLM -> TTS -> Audio
"""
import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, JSONResponse
import uvicorn
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-realtime")
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Jarvis Real-time Voice Server")

# Config
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")  # e.g., https://xxx.ngrok-free.app
VOBIZ_AUTH_ID = os.getenv("VOBIZ_AUTH_ID", "MA_0D6NBKHU")

# Audio config for Vobiz Stream
# Vobiz sends: audio/x-mulaw;rate=8000
SAMPLE_RATE = 8000

# Conversation state per call
conversations = {}


def get_public_ws_url() -> str:
    """Get WebSocket URL for Vobiz Stream verb."""
    if PUBLIC_URL:
        # Convert https:// to wss://
        ws_base = PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/ws"
    return "wss://localhost/ws"


@app.get("/health")
async def health():
    return {"status": "ok", "service": "jarvis-realtime"}


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(request: Request):
    """
    Vobiz calls this when the call is answered.
    Returns XML with <Stream> for bidirectional audio.
    """
    ws_url = get_public_ws_url()
    logger.info(f"Call answered, streaming to: {ws_url}")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" audioTrack="inbound" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">
        {ws_url}
    </Stream>
</Response>"""

    return Response(content=xml, media_type="text/xml")


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Convert mulaw 8kHz to PCM16 8kHz."""
    import audioop
    return audioop.ulaw2lin(mulaw_bytes, 2)


def pcm16_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Convert PCM16 8kHz to mulaw 8kHz."""
    import audioop
    return audioop.lin2ulaw(pcm_bytes, 2)


async def transcribe_with_sarvam(audio_pcm16: bytes) -> str:
    """
    Transcribe PCM16 8kHz audio using Sarvam STT.
    Returns transcribed text.
    """
    if not SARVAM_API_KEY:
        logger.warning("SARVAM_API_KEY not set, STT skipped")
        return ""

    # Sarvam STT expects WAV file
    # Create WAV in memory: 8kHz, 16-bit, mono
    import wave
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(audio_pcm16)
    wav_buffer.seek(0)

    # Call Sarvam API
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
            data = {
                "model": "saarika:v2.5",
                "language_code": "hi-IN",  # Hindi + English
            }
            headers = {"api-subscription-key": SARVAM_API_KEY}

            resp = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                files=files,
                data=data,
                headers=headers,
            )
            if resp.status_code == 200:
                result = resp.json()
                text = result.get("transcript", "")
                logger.info("STT succeeded")
                return text
            else:
                logger.error(f"Sarvam STT failed: {resp.status_code} {resp.text[:200]}")
                return ""
    except Exception as e:
        logger.error(f"STT error: {e}")
        return ""


async def generate_sarvam_tts(text: str, output_path: str) -> bool:
    """
    Generate speech using Sarvam TTS (bulbul:v3).
    Returns True on success.
    """
    if not SARVAM_API_KEY:
        logger.warning("SARVAM_API_KEY not set, TTS skipped")
        return False

    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Sarvam TTS API
            data = {
                "text": text,
                "target_language_code": "hi-IN",
                "speaker": "kabir",  # More natural, engaging voice
                "model": "bulbul:v3",
            }
            headers = {
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            }

            resp = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                json=data,
                headers=headers,
            )
            if resp.status_code == 200:
                result = resp.json()
                # Sarvam returns base64 audio
                audio_b64 = result.get("audios", [None])[0]
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)
                    size = os.path.getsize(output_path)
                    logger.info(f"Sarvam TTS generated: {size} bytes")
                    return size > 1000
                else:
                    logger.error("Sarvam TTS: no audio in response")
                    return False
            else:
                logger.error(f"Sarvam TTS failed: {resp.status_code} {resp.text[:200]}")
                return False
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False


def generate_magnus_tts(text: str, output_path: str) -> bool:
    """
    DEPRECATED: Magnus voice only works in Muse workspace.
    Use generate_sarvam_tts instead for deployable TTS.
    """
    logger.warning("Magnus TTS not available outside workspace, using Sarvam")
    return False


def mp3_to_mulaw8k(mp3_path: str) -> bytes:
    """
    Convert MP3 to mulaw 8kHz bytes for Vobiz Stream.
    Uses ffmpeg.
    """
    try:
        # Convert MP3 -> raw mulaw 8kHz mono
        cmd = [
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", "8000", "-ac", "1",
            "-f", "mulaw", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0:
            return result.stdout
        else:
            logger.error(f"ffmpeg failed: {result.stderr.decode()[:200]}")
            return b""
    except Exception as e:
        logger.error(f"Audio convert error: {e}")
        return b""


# Cached BTC price (updated in background, never blocks)
_btc_price_cache = "unavailable"
_btc_last_fetch = 0

async def get_btc_price() -> str:
    """Get cached BTC price (fast, never blocks). Updates in background every 60s."""
    global _btc_price_cache, _btc_last_fetch
    import time
    now = time.time()
    # Update cache in background if stale (don't await)
    if now - _btc_last_fetch > 60:
        _btc_last_fetch = now
        asyncio.create_task(_fetch_btc_price())
    return _btc_price_cache

async def _fetch_btc_price():
    """Background task to update BTC price cache."""
    global _btc_price_cache
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr"
            )
            if resp.status_code == 200:
                data = resp.json()
                price = data.get("bitcoin", {}).get("inr", 0)
                if price:
                    _btc_price_cache = f"₹{price:,.0f}"
    except:
        pass


async def ask_gemini(user_text: str, history: list) -> str:
    """
    Call Google Gemini API (free tier) for natural conversation.
    Returns the assistant's reply text.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set, LLM skipped")
        return ""

    import httpx
    try:
        # Fetch live BTC price for context
        btc_price = await get_btc_price()

        # Build conversation context
        system_prompt = (
            "You are Jarvis, Harshit Singh's personal voice assistant. "
            "You speak Hindi, English, and Hinglish naturally. "
            "Keep responses SHORT (1-2 sentences max) for voice - they will be spoken aloud. "
            "Be warm, helpful, and a bit playful. "
            "Do not assume current personal, financial or account facts. "
            f"Current live BTC price: {btc_price} INR. "
            "Use this when he asks about Bitcoin or crypto. "
            "Never mention you are an AI model. You are Jarvis."
        )

        # Build contents with history
        contents = []
        for msg in history[-8:]:  # Last 8 messages for context
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": user_text}]
        })

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": 100,  # Keep short for voice
                        "temperature": 0.7,
                    },
                },
            )
            if resp.status_code == 200:
                result = resp.json()
                try:
                    candidates = result.get("candidates", [])
                    if not candidates:
                        logger.warning("Gemini: empty candidates")
                        return ""
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if not parts:
                        logger.warning("Gemini: empty parts")
                        return ""
                    text = parts[0].get("text", "").strip()
                    if not text:
                        logger.warning("Gemini: empty text")
                        return ""
                    logger.info("Gemini succeeded")
                    return text
                except (KeyError, IndexError, AttributeError) as e:
                    logger.error(f"Gemini parse error: {e}, response: {str(result)[:200]}")
                    return ""
            else:
                logger.error(f"Gemini failed: {resp.status_code} {resp.text[:200]}")
                return ""
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return ""


def process_command(text: str, call_id: str) -> str:
    """
    Process user speech and generate response.
    This is where Muse LLM integration goes.
    For now: simple rule-based + personalized responses.
    """
    text_lower = text.lower().strip()

    # Get conversation history
    history = conversations.get(call_id, [])
    history.append({"role": "user", "content": text})

    # Simple responses (will be replaced with full LLM)
    response = ""
    if any(w in text_lower for w in ["hello", "namaste", "sat sri", "hey"]):
        response = "Namaste Harshit! Main Jarvis hun. Boliye, kya karna hai?"
    elif any(w in text_lower for w in ["time", "samay", "baj"]):
        from datetime import datetime
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        response = f"Abhi {now.strftime('%I:%M %p')} ho rahe hain."
    elif any(w in text_lower for w in ["bitcoin", "btc", "crypto"]):
        response = "Mere paas aapki current Bitcoin position ya live market data verify karne ka access nahi hai."
    elif any(w in text_lower for w in ["bye", "alvida", "rakh", "cut"]):
        response = "Theek hai Harshit, phir baat karte hain. Bye!"
    elif text_lower:
        # Default: be honest, don't promise updates we can't deliver
        response = f"Aapne kaha: {text}. Iske baare me mere paas abhi live data nahi hai."
    else:
        response = "Sunai nahi diya, phir se boliye?"

    history.append({"role": "assistant", "content": response})
    conversations[call_id] = history

    return response


async def ask_gemini_stream(user_text: str, history: list):
    """
    Ask Gemini using streaming - yields text chunks as they arrive.
    This enables sentence-by-sentence TTS for lower perceived latency.
    """
    import re
    try:
        # Fetch live BTC price for context
        btc_price = await get_btc_price()

        # Build system prompt
        system_prompt = (
            "You are Jarvis, Harshit Singh's personal voice assistant. "
            "You speak Hindi, English, and Hinglish naturally. "
            "Keep responses SHORT (1-2 sentences max) for voice - they will be spoken aloud. "
            "Be warm, helpful, and a bit playful. "
            "Do not assume current personal, financial or account facts. "
            f"Current live BTC price: {btc_price} INR. "
            "Use this when he asks about Bitcoin or crypto. "
            "Never mention you are an AI model. You are Jarvis."
        )

        # Build contents with history
        contents = []
        for msg in history[-8:]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": user_text}]
        })

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:streamGenerateContent?alt=sse",
                headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": 100,
                        "temperature": 0.7,
                    },
                },
            ) as resp:
                if resp.status_code != 200:
                    logger.error(f"Gemini stream failed: {resp.status_code}")
                    return

                buffer = ""
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                chunk = parts[0].get("text", "")
                                if chunk:
                                    buffer += chunk
                                    # Yield complete sentences
                                    sentences = re.split(r'(?<=[.!?])\s+', buffer)
                                    if len(sentences) > 1:
                                        for s in sentences[:-1]:
                                            if s.strip():
                                                yield s.strip()
                                        buffer = sentences[-1]
                    except:
                        continue

                # Yield remaining buffer
                if buffer.strip():
                    yield buffer.strip()

    except Exception as e:
        logger.error(f"Gemini stream error: {e}")
        return


async def process_command_llm(text: str, call_id: str) -> str:
    """
    Process user speech using Google Gemini LLM with conversation history.
    Falls back to rule-based if LLM fails.
    """
    # Get conversation history
    history = conversations.get(call_id, [])

    # Try Gemini LLM
    response = await ask_gemini(text, history)
    if response:
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})
        conversations[call_id] = history
        return response

    # Fallback to rule-based
    logger.warning("Gemini failed, using rule-based fallback")
    return process_command(text, call_id)


def is_speech(pcm16_bytes: bytes, threshold: int = 500) -> bool:
    """
    Simple energy-based Voice Activity Detection.
    Returns True if the audio frame contains speech (not silence).
    """
    import audioop
    if not pcm16_bytes:
        return False
    # Calculate RMS energy of the audio frame
    try:
        rms = audioop.rms(pcm16_bytes, 2)  # 2 = 16-bit samples
        return rms > threshold
    except:
        return False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bidirectional audio WebSocket for Vobiz Stream.
    Vobiz sends: start (with nested streamId/callId), media, playedStream, clearedAudio
    We send back: playAudio, checkpoint, clearAudio, stop
    """
    await websocket.accept()
    call_id = f"call_{int(time.time())}"
    conversations[call_id] = []
    stream_id = None

    logger.info(f"WebSocket connected: {call_id}")

    # Audio buffer for incoming speech
    audio_buffer = bytearray()
    last_audio_time = time.time()
    is_speaking = False

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                # IDs are NESTED: data.start.streamId, data.start.callId
                start_data = data.get("start", {})
                stream_id = start_data.get("streamId")
                vobiz_call_id = start_data.get("callId")
                logger.info(f"Stream started: streamId={stream_id}, callId={vobiz_call_id}")
                # Send greeting AFTER start event (Vobiz is ready now)
                greeting = "Namaste Harshit! Main Jarvis hun, aapka personal assistant. Boliye, kya karna hai?"
                await speak_text(websocket, greeting, call_id, stream_id)
                continue

            elif event == "media":
                # Incoming audio from caller (base64 mulaw 8kHz)
                payload = data.get("media", {}).get("payload", "")
                if payload:
                    mulaw_bytes = base64.b64decode(payload)
                    pcm16 = mulaw_to_pcm16(mulaw_bytes)
                    # Only buffer and update speech timer if actual speech detected
                    # (Vobiz sends media continuously, even during silence)
                    if is_speech(pcm16):
                        audio_buffer.extend(pcm16)
                        last_audio_time = time.time()
                        is_speaking = True

            elif event == "playedStream":
                logger.info(f"Audio played: {data.get('name')}")
                continue

            elif event == "clearedAudio":
                logger.info("Audio cleared")
                continue

            # Check for end of speech (0.3 sec silence for low latency)
            if is_speaking and len(audio_buffer) > 8000:  # >0.5 sec of audio
                silence_duration = time.time() - last_audio_time
                if silence_duration > 0.3:  # 0.3 sec silence = end of utterance
                    # Process the buffered audio
                    logger.info(f"Processing {len(audio_buffer)} bytes of audio")
                    pcm_bytes = bytes(audio_buffer)
                    audio_buffer.clear()
                    is_speaking = False

                    # STT
                    text = await transcribe_with_sarvam(pcm_bytes)
                    if text:
                        # Streaming LLM -> Speak sentences as they arrive (low latency)
                        history = conversations.get(call_id, [])
                        full_response = ""
                        sentence_count = 0
                        async for sentence in ask_gemini_stream(text, history):
                            if sentence:
                                full_response += sentence + " "
                                sentence_count += 1
                                logger.info(f"Streaming sentence {sentence_count}")
                                await speak_text(websocket, sentence, call_id, stream_id)

                        # Update conversation history
                        if full_response:
                            history.append({"role": "user", "content": text})
                            history.append({"role": "assistant", "content": full_response.strip()})
                            conversations[call_id] = history
                        else:
                            # Fallback if streaming failed
                            logger.warning("Gemini stream empty, using fallback")
                            response = process_command(text, call_id)
                            await speak_text(websocket, response, call_id, stream_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {call_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if call_id in conversations:
            del conversations[call_id]


async def speak_text(websocket: WebSocket, text: str, call_id: str, stream_id: str = None):
    """Generate Sarvam TTS and stream to caller via WebSocket using playAudio."""
    if not stream_id:
        logger.warning("Playback skipped: stream ID missing")
        return
    logger.info("Generating voice reply")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        mp3_path = tf.name

    try:
        # Generate Sarvam audio (async)
        if not await generate_sarvam_tts(text, mp3_path):
            logger.error("TTS generation failed")
            return

        # Convert to mulaw 8kHz
        mulaw_bytes = mp3_to_mulaw8k(mp3_path)
        if not mulaw_bytes:
            logger.error("Audio conversion failed")
            return

        # Stream in chunks (20ms = 160 bytes at 8kHz mulaw)
        # Vobiz expects: {event: "playAudio", media: {contentType, sampleRate, payload}}
        chunk_size = 160
        for i in range(0, len(mulaw_bytes), chunk_size):
            chunk = mulaw_bytes[i:i+chunk_size]
            payload = base64.b64encode(chunk).decode()

            msg = {
                "event": "playAudio",
                "streamId": stream_id,
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "payload": payload
                }
            }
            await websocket.send_text(json.dumps(msg))

            # Real-time pacing: 20ms per chunk
            await asyncio.sleep(0.02)

        # Send checkpoint to mark end of utterance
        if stream_id:
            checkpoint_msg = {
                "event": "checkpoint",
                "streamId": stream_id,
                "name": f"utterance_{int(time.time())}"
            }
            await websocket.send_text(json.dumps(checkpoint_msg))

        logger.info(f"Finished speaking ({len(mulaw_bytes)} bytes)")

    finally:
        if os.path.exists(mp3_path):
            os.unlink(mp3_path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
