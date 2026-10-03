"""
Jarvis Live Voice Server (Gemini Live API - Native Audio)
WebSocket-based bidirectional audio for Vobiz <Stream>
Architecture: Vobiz audio -> Gemini Live API (native audio) -> Vobiz audio
No STT/TTS pipeline - Gemini handles audio natively for minimal latency.
"""
import asyncio
import base64
import json
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
import uvicorn
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-live")

app = FastAPI(title="Jarvis Live Voice Server (Gemini Live API)")

# Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
VOBIZ_AUTH_ID = os.getenv("VOBIZ_AUTH_ID", "MA_0D6NBKHU")

# Gemini Live API config
GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"
GEMINI_LIVE_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# Audio config
VOBIZ_SAMPLE_RATE = 8000   # Vobiz uses mulaw 8kHz
GEMINI_INPUT_RATE = 16000  # Gemini Live expects PCM16 16kHz input
GEMINI_OUTPUT_RATE = 24000  # Gemini Live outputs PCM16 24kHz

# Conversation state per call (for logging/transcripts)
conversations = {}

# Cached BTC price (updated in background)
_btc_price_cache = "unavailable"
_btc_last_fetch = 0


def get_public_ws_url() -> str:
    """Get WebSocket URL for Vobiz Stream verb."""
    if PUBLIC_URL:
        ws_base = PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/ws"
    return "wss://localhost/ws"


@app.get("/health")
async def health():
    return {"status": "ok", "service": "jarvis-live"}


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(request: Request):
    """Vobiz calls this when the call is answered. Returns Stream XML."""
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


def upsample_8k_to_16k(pcm8k: bytes) -> bytes:
    """Upsample PCM16 8kHz mono to PCM16 16kHz mono."""
    import audioop
    out, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    return out


def downsample_24k_to_8k(pcm24k: bytes) -> bytes:
    """Downsample PCM16 24kHz mono to PCM16 8kHz mono."""
    import audioop
    out, _ = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, None)
    return out


async def _fetch_btc_price():
    """Background task to update BTC price cache."""
    global _btc_price_cache
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr"
            )
            if resp.status_code == 200:
                data = resp.json()
                price = data.get("bitcoin", {}).get("inr", 0)
                if price:
                    _btc_price_cache = f"₹{price:,.0f}"
                    logger.info(f"BTC price updated: {_btc_price_cache}")
    except Exception as e:
        logger.warning(f"BTC price fetch failed: {e}")


async def get_btc_price() -> str:
    """Get cached BTC price (fast, never blocks). Refresh in background if stale."""
    global _btc_last_fetch
    now = time.time()
    if now - _btc_last_fetch > 60:
        _btc_last_fetch = now
        asyncio.create_task(_fetch_btc_price())
    return _btc_price_cache


def build_system_instruction(btc_price: str) -> str:
    return (
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


class GeminiLiveSession:
    """
    Manages a Gemini Live API WebSocket session.
    Forwards Vobiz audio to Gemini, forwards Gemini audio back to Vobiz.
    """

    def __init__(self, vobiz_ws: WebSocket, call_id: str, stream_id: Optional[str]):
        self.vobiz_ws = vobiz_ws
        self.call_id = call_id
        self.stream_id = stream_id
        self.gemini_ws = None
        self.running = False
        self._recv_task = None
        self._send_lock = asyncio.Lock()
        # Buffer for outgoing audio pacing
        self._audio_queue = asyncio.Queue()

    async def connect(self) -> bool:
        """Connect to Gemini Live API and send setup."""
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set, cannot connect to Gemini Live")
            return False

        try:
            url = f"{GEMINI_LIVE_WS_URL}?key={GEMINI_API_KEY}"
            self.gemini_ws = await websockets.connect(
                url,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            )
            logger.info("Connected to Gemini Live API")

            # Build setup message (must be first message)
            btc_price = await get_btc_price()
            setup_msg = {
                "setup": {
                    "model": f"models/{GEMINI_LIVE_MODEL}",
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": "Aoede"
                                }
                            }
                        }
                    },
                    "systemInstruction": {
                        "parts": [{"text": build_system_instruction(btc_price)}]
                    },
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {},
                    "realtimeInputConfig": {
                        "automaticActivityDetection": {
                            "disabled": False
                        }
                    }
                }
            }
            await self.gemini_ws.send(json.dumps(setup_msg))
            logger.info("Gemini Live setup sent, waiting for confirmation...")

            # Wait for setupComplete
            async with asyncio.timeout(15):
                async for msg in self.gemini_ws:
                    data = json.loads(msg)
                    if "setupComplete" in data:
                        logger.info("Gemini Live setup complete")
                        break
                    # Handle any early messages
                    logger.debug(f"Pre-setup message: {str(data)[:100]}")

            self.running = True
            self._recv_task = asyncio.create_task(self._receive_loop())
            return True

        except Exception as e:
            logger.error(f"Gemini Live connect failed: {e}")
            return False

    async def _receive_loop(self):
        """Receive messages from Gemini Live API and forward audio to Vobiz."""
        try:
            async for msg in self.gemini_ws:
                if not self.running:
                    break
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue

                # Handle server content (audio output, transcriptions)
                server_content = data.get("serverContent")
                if server_content:
                    # Audio output
                    model_turn = server_content.get("modelTurn", {})
                    for part in model_turn.get("parts", []):
                        inline_data = part.get("inlineData", {})
                        if inline_data and inline_data.get("mimeType", "").startswith("audio/"):
                            audio_b64 = inline_data.get("data", "")
                            if audio_b64:
                                pcm24k = base64.b64decode(audio_b64)
                                await self._forward_audio_to_vobiz(pcm24k)

                    # Transcriptions (for logging)
                    input_trans = server_content.get("inputTranscription", {})
                    if input_trans.get("text"):
                        logger.info(f"User said: {input_trans['text']}")

                    output_trans = server_content.get("outputTranscription", {})
                    if output_trans.get("text"):
                        logger.info(f"Jarvis said: {output_trans['text']}")

                    # Turn complete
                    if server_content.get("turnComplete"):
                        logger.debug("Turn complete")

                # Handle interruption (user started speaking)
                if data.get("serverContent", {}).get("interrupted"):
                    logger.info("Gemini interrupted (barge-in detected)")
                    # Clear Vobiz audio buffer so stale audio doesn't play
                    await self._clear_vobiz_audio()

                # Handle GoAway (server asking us to reconnect)
                if "goAway" in data:
                    logger.warning(f"Gemini sent GoAway: {data['goAway']}")
                    break

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Gemini Live connection closed: {e}")
        except Exception as e:
            logger.error(f"Gemini Live receive loop error: {e}")
        finally:
            self.running = False

    async def _forward_audio_to_vobiz(self, pcm24k: bytes):
        """Convert Gemini PCM24k audio to mulaw 8kHz and stream to Vobiz."""
        try:
            # Downsample 24kHz -> 8kHz, then convert to mulaw
            pcm8k = downsample_24k_to_8k(pcm24k)
            mulaw = pcm16_to_mulaw(pcm8k)

            # Stream in 20ms chunks (160 bytes at 8kHz mulaw)
            chunk_size = 160
            for i in range(0, len(mulaw), chunk_size):
                if not self.running:
                    break
                chunk = mulaw[i:i + chunk_size]
                payload = base64.b64encode(chunk).decode()
                msg = {
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "payload": payload
                    }
                }
                async with self._send_lock:
                    await self.vobiz_ws.send_text(json.dumps(msg))
                # Real-time pacing
                await asyncio.sleep(0.02)

        except Exception as e:
            logger.error(f"Error forwarding audio to Vobiz: {e}")

    async def _clear_vobiz_audio(self):
        """Send clearAudio to Vobiz to stop stale playback on barge-in."""
        try:
            msg = {
                "event": "clearAudio",
                "streamId": self.stream_id,
            }
            async with self._send_lock:
                await self.vobiz_ws.send_text(json.dumps(msg))
            logger.info("Sent clearAudio to Vobiz (barge-in)")
        except Exception as e:
            logger.error(f"Error sending clearAudio: {e}")

    async def send_audio(self, pcm8k: bytes):
        """Forward Vobiz PCM16 8kHz audio to Gemini Live API (upsampled to 16kHz)."""
        if not self.gemini_ws or not self.running:
            return
        try:
            # Upsample 8kHz -> 16kHz for Gemini
            pcm16k = upsample_8k_to_16k(pcm8k)
            audio_b64 = base64.b64encode(pcm16k).decode()
            msg = {
                "realtimeInput": {
                    "mediaChunks": [
                        {
                            "mimeType": "audio/pcm;rate=16000",
                            "data": audio_b64
                        }
                    ]
                }
            }
            await self.gemini_ws.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")

    async def send_greeting(self, text: str):
        """Send initial text greeting - Gemini will speak it."""
        if not self.gemini_ws or not self.running:
            return
        try:
            msg = {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [{"text": f"Please greet me with: {text}"}]
                        }
                    ],
                    "turnComplete": True
                }
            }
            await self.gemini_ws.send(json.dumps(msg))
            logger.info("Greeting sent to Gemini Live")
        except Exception as e:
            logger.error(f"Error sending greeting: {e}")

    async def close(self):
        """Close the Gemini Live session."""
        self.running = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self.gemini_ws:
            try:
                await self.gemini_ws.close()
            except:
                pass
        logger.info("Gemini Live session closed")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bidirectional audio WebSocket for Vobiz Stream.
    Forwards audio between Vobiz and Gemini Live API (native audio, no STT/TTS).
    """
    await websocket.accept()
    call_id = f"call_{int(time.time())}"
    conversations[call_id] = []
    stream_id = None

    logger.info(f"WebSocket connected: {call_id}")

    # Create Gemini Live session
    session = GeminiLiveSession(websocket, call_id, stream_id)

    try:
        # Wait for 'start' event to get stream_id
        message = await websocket.receive_text()
        data = json.loads(message)
        if data.get("event") == "start":
            start_data = data.get("start", {})
            stream_id = start_data.get("streamId")
            session.stream_id = stream_id
            vobiz_call_id = start_data.get("callId")
            logger.info(f"Stream started: streamId={stream_id}, callId={vobiz_call_id}")

        # Connect to Gemini Live API
        if not await session.connect():
            logger.error("Failed to connect to Gemini Live API, closing call")
            await websocket.close()
            return

        # Send greeting (Gemini will speak it natively)
        greeting = "Namaste Harshit! Main Jarvis hun, aapka personal assistant. Boliye, kya karna hai?"
        await session.send_greeting(greeting)

        # Audio forwarding loop: Vobiz -> Gemini
        # Buffer small chunks to reduce WebSocket overhead (~100ms = 1600 bytes PCM16 8kHz)
        audio_buffer = bytearray()
        BUFFER_SIZE = 1600  # 100ms of PCM16 8kHz audio

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")

            if event == "media":
                payload = data.get("media", {}).get("payload", "")
                if payload:
                    mulaw_bytes = base64.b64decode(payload)
                    pcm16_8k = mulaw_to_pcm16(mulaw_bytes)
                    audio_buffer.extend(pcm16_8k)

                    # Forward buffered audio to Gemini when we have enough
                    while len(audio_buffer) >= BUFFER_SIZE:
                        chunk = bytes(audio_buffer[:BUFFER_SIZE])
                        del audio_buffer[:BUFFER_SIZE]
                        await session.send_audio(chunk)

            elif event == "playedStream":
                logger.debug(f"Audio played: {data.get('name')}")
                continue

            elif event == "clearedAudio":
                logger.info("Audio cleared by Vobiz")
                continue

            elif event == "stop":
                logger.info("Vobiz sent stop event")
                break

        # Flush remaining buffered audio
        if audio_buffer:
            await session.send_audio(bytes(audio_buffer))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {call_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await session.close()
        if call_id in conversations:
            del conversations[call_id]


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
