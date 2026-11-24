import { useEffect, useRef, forwardRef, useImperativeHandle, useState } from "react";
import { SimliClient, LogLevel } from "simli-client";

const API_BASE = "http://localhost:8000";

// Exactly 6KB chunks, per CTO recommendation
const CHUNK_SIZE = 6144; 

function describeError(err) {
  const msg = String(err?.message || err);
  if (msg.includes("402")) return "Avatar unavailable: Simli credits are used up.";
  if (msg.includes("401") || msg.includes("403")) return "Avatar unavailable: check SIMLI_API_KEY.";
  if (msg.includes("Failed to fetch")) return "Avatar unavailable: backend is not running.";
  return "Avatar unavailable right now.";
}


const AvatarSimli = forwardRef((props, ref) => {
  const videoRef = useRef(null);
  const audioRef = useRef(null);
  const simliClientRef = useRef(null);
  const connectingRef = useRef(null);
  
  const [status, setStatus] = useState("connecting"); 
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const [errorText, setErrorText] = useState("");

  const closeStaleClient = () => {
    simliClientRef.current?.close?.();
    simliClientRef.current = null;
  };

  const connectFresh = async () => {
    setStatus("connecting");
    const res = await fetch(`${API_BASE}/simli-session`);
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(`simli-session ${res.status}: ${detail.detail || ""}`);
    }
    const { sessionToken, iceServers } = await res.json();

    const client = new SimliClient(
      sessionToken,
      videoRef.current,
      audioRef.current,
      iceServers,
      LogLevel.INFO,
      "p2p"
    );

    client.on("error", (detail) => {
      console.error("Simli error event:", detail);
      setStatus("error");
    });
    
    client.on("startup_error", (message) => {
      console.error("Simli startup_error event:", message);
      setStatus("error");
    });
    
    client.on("stop", () => {
      console.warn("Simli: connection stopped");
      simliClientRef.current = null;
      setStatus("idle"); 
    });

    simliClientRef.current = client;
    await client.start();
    
    setStatus("ready");
    setErrorText("");
    setHasLoadedOnce(true);
    return client;
  };

  const ensureConnected = async () => {
    if (simliClientRef.current) return simliClientRef.current;
    if (connectingRef.current) return connectingRef.current;

    connectingRef.current = connectFresh();
    try {
      return await connectingRef.current;
    } catch (err) {
      console.error("Simli session failed to start:", err);
      setStatus("error");
      simliClientRef.current = null;
      throw err;
    } finally {
      connectingRef.current = null;
    }
  };

  useEffect(() => {
    // Show why Simli couldn't start (bad key, no credits, network) instead
    // of an endless "Connecting…"; chat keeps working meanwhile.
    ensureConnected().catch((err) => setErrorText(describeError(err)));
    return () => closeStaleClient();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useImperativeHandle(ref, () => ({
    /**
     * Play a streamed TTS response (raw 16 kHz mono PCM16). Audio is pushed
     * to Simli chunk by chunk as it downloads, so the avatar starts talking
     * as soon as the first bytes arrive instead of after the whole reply.
     * @param {Response} response  fetch() response from POST /tts
     * @param {AbortSignal} signal  aborted when a newer reply starts
     */
    speakStream: async (response, signal) => {
      let client;
      try {
        client = await ensureConnected();
      } catch {
        return;
      }
      client.ClearBuffer?.(); // stop whatever was still being said

      const reader = response.body.getReader();
      let pending = new Uint8Array(0);
      try {
        while (!signal?.aborted) {
          const { done, value } = await reader.read();
          if (done) break;
          // join with leftover bytes so we never split a 16-bit sample
          const merged = new Uint8Array(pending.length + value.length);
          merged.set(pending);
          merged.set(value, pending.length);
          const usable = merged.length - (merged.length % 2);
          for (let offset = 0; offset < usable; offset += CHUNK_SIZE) {
            client.sendAudioData(merged.slice(offset, Math.min(offset + CHUNK_SIZE, usable)));
          }
          pending = merged.slice(usable);
        }
      } catch (err) {
        if (signal?.aborted) return;
        console.error("Streaming audio to Simli failed, discarding session:", err);
        closeStaleClient();
        setStatus("error");
      } finally {
        if (signal?.aborted) reader.cancel().catch(() => {});
      }
    },
  }));

  return (
    <div style={{ width: "100%", height: "100%", minHeight: "350px", position: "relative", borderRadius: "12px", overflow: "hidden", background: "linear-gradient(135deg, #1a2a3a, #0a0a0a)" }}>
      <video ref={videoRef} autoPlay playsInline poster="/face.png" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      <audio ref={audioRef} autoPlay />
      
      {!hasLoadedOnce && status !== "ready" && (
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "12px", zIndex: 10, background: "#111" }}>
          <img src="/logo.png" alt="" style={{ width: "56px", height: "56px", opacity: 0.6 }} />
          <div style={{ color: "#cfd8e3", fontSize: "0.9rem" }}>
            {errorText || (status === "error" ? "Reconnecting…" : "Connecting to Receptionist…")}
          </div>
        </div>
      )}
    </div>
  );
});

export default AvatarSimli;