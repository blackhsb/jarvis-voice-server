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
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-realtime")

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
                logger.info(f"STT: {text}")
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


async def get_btc_price() -> str:
    """Fetch live BTC/INR price from CoinGecko."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr"
            )
            if resp.status_code == 200:
                data = resp.json()
                price = data.get("bitcoin", {}).get("inr", 0)
                if price:
                    return f"₹{price:,.0f}"
    except:
        pass
    return "unavailable"


def pcm16_to_wav(pcm16_bytes: bytes, sample_rate: int = 8000) -> bytes:
    """
    Wrap raw PCM16 mono bytes in a minimal WAV header.
    Sarvam streaming STT expects 'audio/wav' encoding per message.
    """
    import struct
    n = len(pcm16_bytes)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + n, b'WAVE',
        b'fmt ', 16,
        1, 1, sample_rate,
        sample_rate * 2, 2, 16,
        b'data', n,
    )
    return header + pcm16_bytes


class SarvamStreamingSTT:
    """
    Persistent WebSocket connection to Sarvam streaming STT (saaras:v3).
    Streams WAV-wrapped PCM16 audio chunks; yields transcripts + VAD events.
    Protocol reverse-engineered from the official sarvamai SDK (v0.1.35).
    Falls back to batch STT if the stream cannot be established.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws = None
        self.msg_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self._listen_task = None
        self._send_lock = asyncio.Lock()

    async def _connect_ws(self):
        """Establish the raw WebSocket (used by connect + reconnect)."""
        url = (
            "wss://api.sarvam.ai/speech-to-text/ws"
            "?model=saaras:v3"
            "&mode=transcribe"
            "&language_code=hi-IN"
            "&sample_rate=8000"
            "&vad_signals=true"
            "&high_vad_sensitivity=true"
        )
        self.ws = await websockets.connect(
            url,
            extra_headers={"Api-Subscription-Key": self.api_key},
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )

    async def connect(self) -> bool:
        """Connect to Sarvam streaming STT WebSocket."""
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not set, streaming STT disabled")
            return False
        try:
            await self._connect_ws()
            self.running = True
            self._listen_task = asyncio.create_task(self._listen())
            logger.info("Sarvam streaming STT connected")
            return True
        except Exception as e:
            logger.error(f"Sarvam streaming STT connect failed: {e}")
            return False

    async def _listen(self):
        """Listen for transcripts / VAD events; reconnect on drops."""
        backoff = 1
        while self.running:
            try:
                async for msg in self.ws:
                    backoff = 1  # reset on successful message
                    try:
                        data = json.loads(msg) if isinstance(msg, str) else msg
                        await self.msg_queue.put(data)
                    except Exception:
                        continue
                # Server closed cleanly; try to reconnect
                logger.warning("STT stream closed by server, reconnecting...")
                self.ws = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"STT listen error: {e}")
                self.ws = None
            if not self.running:
                break
            try:
                await asyncio.sleep(backoff)
                await self._connect_ws()
                logger.info("Sarvam streaming STT reconnected")
                backoff = 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"STT reconnect failed: {e}")
                backoff = min(backoff * 2, 30)
        self.running = False

    async def send_audio_chunk(self, pcm16_bytes: bytes):
        """Send one PCM16 chunk (wrapped as WAV) to Sarvam."""
        if not self.ws or not self.running or not pcm16_bytes:
            return
        try:
            wav = pcm16_to_wav(pcm16_bytes, 8000)
            b64 = base64.b64encode(wav).decode("ascii")
            async with self._send_lock:
                await self.ws.send(json.dumps({
                    "audio": {
                        "data": b64,
                        "sample_rate": 8000,
                        "encoding": "audio/wav",
                    }
                }))
        except Exception as e:
            logger.error(f"STT send audio failed: {e}")

    async def flush(self):
        """Force Sarvam to finalize any pending partial transcript."""
        if not self.ws or not self.running:
            return
        try:
            async with self._send_lock:
                await self.ws.send(json.dumps({"type": "flush"}))
        except Exception as e:
            logger.error(f"STT flush failed: {e}")

    async def get_message(self, timeout: float = 0.2):
        """Get next raw message (non-blocking with timeout)."""
        try:
            return await asyncio.wait_for(self.msg_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self):
        """Close the connection."""
        self.running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        logger.info("Sarvam streaming STT closed")


class SarvamStreamingTTS:
    """
    Persistent WebSocket connection to Sarvam streaming TTS (bulbul:v3).
    Sends text, receives base64 MP3 chunks, decodes to mulaw 8kHz in
    real time via a piped ffmpeg process.
    Protocol reverse-engineered from the official sarvamai SDK (v0.1.35).
    """
    def __init__(self, api_key: str, speaker: str = "kabir"):
        self.api_key = api_key
        self.speaker = speaker
        self.ws = None
        self._send_lock = asyncio.Lock()
        self._speak_lock = asyncio.Lock()  # one utterance at a time

    async def connect(self) -> bool:
        """Connect + send initial TTS configuration."""
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not set, streaming TTS disabled")
            return False
        try:
            url = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3"
            self.ws = await websockets.connect(
                url,
                extra_headers={"Api-Subscription-Key": self.api_key},
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            )
            await self.ws.send(json.dumps({
                "type": "config",
                "data": {
                    "language_code": "hi-IN",
                    "speaker": self.speaker,
                    "speech_sample_rate": 24000,
                    "output_audio_codec": "mp3",
                    "min_buffer_size": 30,
                    "pace": 1.0,
                },
            }))
            logger.info("Sarvam streaming TTS connected")
            return True
        except Exception as e:
            logger.error(f"Sarvam streaming TTS connect failed: {e}")
            return False

    async def synthesize_to_mulaw(self, text: str, chunk_callback):
        """
        Synthesize `text`; invoke `chunk_callback(mulaw_bytes)` for each
        decoded 8kHz mulaw chunk as soon as it is available.
        Raises asyncio.CancelledError if the calling task is cancelled
        (used for barge-in).
        """
        if not self.ws:
            raise RuntimeError("TTS WebSocket not connected")
        async with self._speak_lock:
            # ffmpeg: MP3 (stdin pipe) -> mulaw 8kHz mono (stdout pipe)
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-ar", "8000", "-ac", "1", "-f", "mulaw", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            async def pump_stdout():
                try:
                    while True:
                        data = await proc.stdout.read(320)  # 40ms @ 8kHz
                        if not data:
                            break
                        await chunk_callback(data)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass

            pump_task = asyncio.create_task(pump_stdout())
            try:
                async with self._send_lock:
                    await self.ws.send(json.dumps(
                        {"type": "text", "data": {"text": text}}))
                    await self.ws.send(json.dumps({"type": "flush"}))

                done = False
                while not done:
                    raw = await self.ws.recv()
                    msg = json.loads(raw) if isinstance(raw, str) else raw
                    mtype = msg.get("type")
                    if mtype == "audio":
                        b64 = msg.get("data", {}).get("audio", "")
                        if b64:
                            try:
                                proc.stdin.write(base64.b64decode(b64))
                                await proc.stdin.drain()
                            except Exception:
                                break
                    elif mtype == "event":
                        done = True  # synthesis complete
                    elif mtype == "error":
                        logger.error(f"TTS stream error msg: {str(msg)[:200]}")
                        break
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(pump_task, timeout=8.0)
                except Exception:
                    pump_task.cancel()
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        logger.info("Sarvam streaming TTS closed")


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
            "Harshit runs an agency (Blackhsbagency), a men's fashion Instagram (@blackhsbstlyin), "
            "and trades crypto (has 0.000059 BTC position). "
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
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
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
                    logger.info(f"Gemini: {text[:80]}...")
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
        response = "Aapka Bitcoin position khula hai. 0.000059 BTC, entry 84 lakh 3 hazaar. Stop 82 lakh 30 hazaar, target 86 lakh 90 hazaar."
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
            "Harshit runs an agency (Blackhsbagency), a men's fashion Instagram (@blackhsbstlyin), "
            "and trades crypto (has 0.000059 BTC position). "
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
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse",
                headers={"Content-Type": "application/json"},
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
    Bidirectional audio WebSocket for Vobiz Stream - Gemini Live quality.

    Architecture:
    - Inbound audio -> Sarvam streaming STT (real-time transcripts + VAD)
    - Transcripts -> Gemini streaming LLM (sentence-by-sentence)
    - Sentences -> Sarvam streaming TTS (MP3 chunks -> ffmpeg -> mulaw 8kHz)
    - Barge-in: user speech during AI playback cancels TTS immediately
    - Full-duplex: inbound audio processed while outbound plays
    - Fallbacks: batch STT/TTS if streaming unavailable (existing working code)

    Vobiz events in:  start, media, playedStream, clearedAudio
    Vobiz events out: playAudio, checkpoint, clearAudio
    """
    await websocket.accept()
    call_id = f"call_{int(time.time())}"
    conversations[call_id] = []
    stream_id = None

    logger.info(f"WebSocket connected: {call_id}")

    # ---- Streaming clients ----
    stt = SarvamStreamingSTT(SARVAM_API_KEY)
    tts = SarvamStreamingTTS(SARVAM_API_KEY, speaker="kabir")
    stt_ok = await stt.connect()
    tts_ok = await tts.connect()
    if not stt_ok:
        logger.warning("Streaming STT unavailable -> batch STT fallback")
    if not tts_ok:
        logger.warning("Streaming TTS unavailable -> batch TTS fallback")

    # ---- Per-call state ----
    ai_speaking = False          # True while TTS audio is being played
    speak_cancel = asyncio.Event()  # set on barge-in to stop chunk sender
    speak_task: Optional[asyncio.Task] = None
    turn_seq = 0                 # incremented per user turn / barge-in; stale gens abort

    # Batch-STT fallback state (only used when streaming STT is down)
    audio_buffer = bytearray()
    last_audio_time = time.time()
    is_speaking = False

    async def send_play_audio(mulaw: bytes):
        """Send mulaw 8kHz bytes to Vobiz in 20ms paced chunks."""
        chunk_size = 160  # 20ms @ 8kHz mulaw
        for i in range(0, len(mulaw), chunk_size):
            if speak_cancel.is_set():
                break
            chunk = mulaw[i:i + chunk_size]
            payload = base64.b64encode(chunk).decode("ascii")
            await websocket.send_text(json.dumps({
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "payload": payload,
                },
            }))
            await asyncio.sleep(0.02)  # real-time pacing

    async def speak_streaming(text: str):
        """Speak text via streaming TTS (cancellable). Falls back to batch."""
        nonlocal ai_speaking
        if not text or not text.strip():
            return
        ai_speaking = True
        speak_cancel.clear()
        try:
            if tts_ok:
                async def on_mulaw(chunk: bytes):
                    if not speak_cancel.is_set():
                        await send_play_audio(chunk)
                await tts.synthesize_to_mulaw(text, on_mulaw)
            else:
                # Batch fallback (existing tested path)
                await speak_text(websocket, text, call_id, stream_id)
        except asyncio.CancelledError:
            logger.info("speak_streaming cancelled (barge-in)")
            raise
        except Exception as e:
            logger.error(f"speak_streaming error: {e}, trying batch fallback")
            try:
                await speak_text(websocket, text, call_id, stream_id)
            except Exception as e2:
                logger.error(f"batch fallback also failed: {e2}")
        finally:
            ai_speaking = False
            if stream_id and not speak_cancel.is_set():
                try:
                    await websocket.send_text(json.dumps({
                        "event": "checkpoint",
                        "streamId": stream_id,
                        "name": f"utterance_{int(time.time())}",
                    }))
                except Exception:
                    pass

    async def barge_in():
        """User started speaking while AI was talking: stop playback NOW."""
        nonlocal ai_speaking, speak_task, turn_seq
        if not ai_speaking and (speak_task is None or speak_task.done()):
            return
        turn_seq += 1
        logger.info("Barge-in: interrupting AI speech")
        speak_cancel.set()
        if speak_task and not speak_task.done():
            speak_task.cancel()
            try:
                await speak_task
            except asyncio.CancelledError:
                pass
        speak_task = None
        try:
            await websocket.send_text(json.dumps({
                "event": "clearAudio",
                "streamId": stream_id,
            }))
        except Exception:
            pass
        ai_speaking = False

    async def handle_user_turn(text: str):
        """LLM -> speak for one finalized user utterance."""
        nonlocal speak_task, turn_seq
        text = (text or "").strip()
        if not text:
            return
        my_seq = turn_seq
        logger.info(f"User turn: {text[:80]}")
        history = conversations.get(call_id, [])
        full_response = ""
        try:
            async for sentence in ask_gemini_stream(text, history):
                if my_seq != turn_seq or speak_cancel.is_set():
                    logger.info("Turn superseded, aborting generation")
                    break
                if sentence:
                    full_response += sentence + " "
                    logger.info(f"Speaking sentence: {sentence[:50]}...")
                    speak_task = asyncio.create_task(speak_streaming(sentence))
                    try:
                        await speak_task
                    except asyncio.CancelledError:
                        logger.info("Sentence interrupted by barge-in")
                        break
                    speak_task = None
        except asyncio.CancelledError:
            raise

        if my_seq != turn_seq:
            return  # stale turn after barge-in; do not record
        if full_response.strip():
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": full_response.strip()})
            conversations[call_id] = history
        else:
            logger.warning("Gemini stream empty, using rule-based fallback")
            response = process_command(text, call_id)
            speak_task = asyncio.create_task(speak_streaming(response))
            try:
                await speak_task
            except asyncio.CancelledError:
                pass
            speak_task = None

    async def stt_consumer():
        """
        Background: consume Sarvam streaming STT messages.
        - events/START_SPEECH -> barge-in (user took the floor)
        - events/END_SPEECH   -> finalize turn with latest transcript
        - data                -> latest transcript text
        Exits if the STT stream dies; batch fallback only applies at dial time.
        """
        pending = ""
        while stt.running:
            msg = await stt.get_message(timeout=0.2)
            if msg is None:
                continue
            try:
                mtype = msg.get("type")
                if mtype == "events":
                    sig = msg.get("data", {}).get("signal_type", "")
                    if sig == "START_SPEECH":
                        pending = ""
                        await barge_in()
                    elif sig == "END_SPEECH":
                        if pending.strip():
                            t = pending.strip()
                            pending = ""
                            await handle_user_turn(t)
                elif mtype == "data":
                    t = msg.get("data", {}).get("transcript", "")
                    if t:
                        pending = t
                elif mtype == "error":
                    logger.error(f"STT stream error msg: {str(msg)[:200]}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"stt_consumer error: {e}")

    consumer_task = None
    try:
        if stt_ok:
            consumer_task = asyncio.create_task(stt_consumer())

        # Accumulate inbound PCM to forward to streaming STT in ~200ms chunks
        stt_fwd_buf = bytearray()
        last_fwd = time.time()

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                start_data = data.get("start", {})
                stream_id = start_data.get("streamId")
                vobiz_call_id = start_data.get("callId")
                logger.info(f"Stream started: streamId={stream_id}, callId={vobiz_call_id}")
                greeting = "Namaste Harshit! Main Jarvis hun, aapka personal assistant. Boliye, kya karna hai?"
                speak_task = asyncio.create_task(speak_streaming(greeting))
                continue

            elif event == "media":
                payload = data.get("media", {}).get("payload", "")
                if payload:
                    mulaw_bytes = base64.b64decode(payload)
                    pcm16 = mulaw_to_pcm16(mulaw_bytes)
                    speech = is_speech(pcm16)

                    # Fast-path barge-in via local energy VAD (no network round-trip)
                    if speech and ai_speaking:
                        await barge_in()

                    if stt_ok:
                        # Forward everything (speech + silence) so Sarvam VAD
                        # sees true speech boundaries; chunk to ~200ms.
                        stt_fwd_buf.extend(pcm16)
                        if time.time() - last_fwd >= 0.2 and stt_fwd_buf:
                            await stt.send_audio_chunk(bytes(stt_fwd_buf))
                            stt_fwd_buf.clear()
                            last_fwd = time.time()
                    else:
                        # Batch fallback path (existing tested logic)
                        if speech:
                            audio_buffer.extend(pcm16)
                            last_audio_time = time.time()
                            is_speaking = True

            elif event == "playedStream":
                continue

            elif event == "clearedAudio":
                logger.info("Audio cleared by Vobiz")
                ai_speaking = False
                continue

            # Batch-STT fallback turn detection (only when streaming STT is down)
            if not stt_ok and is_speaking and len(audio_buffer) > 8000:
                if time.time() - last_audio_time > 0.3:
                    pcm_bytes = bytes(audio_buffer)
                    audio_buffer.clear()
                    is_speaking = False
                    text = await transcribe_with_sarvam(pcm_bytes)
                    if text:
                        await handle_user_turn(text)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {call_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        turn_seq += 1
        speak_cancel.set()
        if consumer_task:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        if speak_task and not speak_task.done():
            speak_task.cancel()
            try:
                await speak_task
            except asyncio.CancelledError:
                pass
        try:
            await stt.close()
        except Exception:
            pass
        try:
            await tts.close()
        except Exception:
            pass
        if call_id in conversations:
            del conversations[call_id]


async def speak_text(websocket: WebSocket, text: str, call_id: str, stream_id: str = None):
    """Generate Sarvam TTS and stream to caller via WebSocket using playAudio."""
    logger.info(f"Speaking: {text[:50]}...")

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
