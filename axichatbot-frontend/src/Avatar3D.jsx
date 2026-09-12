import { useEffect, useRef, forwardRef, useImperativeHandle, useState } from "react";
import { SimliClient, LogLevel } from "simli-client";

const API_BASE = "http://localhost:8000";

// Exactly 6KB chunks, per CTO recommendation
const CHUNK_SIZE = 6144; 

function wavBase64ToPcm16(base64Wav) {
  const binary = atob(base64Wav);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const view = new DataView(bytes.buffer);
  let offset = 12;
  while (offset < bytes.length) {
    const chunkId = String.fromCharCode(
      bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
    );
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === "data") {
      return bytes.slice(offset + 8, offset + 8 + chunkSize);
    }
    offset += 8 + chunkSize;
  }
  console.warn("No 'data' chunk found in WAV — falling back to byte 44");
  return bytes.slice(44);
}

// --- NEW FUNCTION: The Silence Trimmer ---
// Scans the PCM audio backwards to find where the actual voice stops,
// stripping out the invisible "room tone" the TTS engine leaves at the end.
function trimTrailingSilence(pcmBytes, threshold = 150) {
  const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength);
  let lastVoiceIndex = 0;
  
  // Scan backwards through the audio (2 bytes per sample for PCM16)
  for (let i = view.byteLength - 2; i >= 0; i -= 2) {
    const sampleAmplitude = Math.abs(view.getInt16(i, true));
    
    // If we hit a sound louder than our silence threshold, mark this as the true end
    if (sampleAmplitude > threshold) {
      lastVoiceIndex = i;
      break;
    }
  }
  
  // Keep exactly 0.1 seconds (3200 bytes) of padding after the voice stops 
  // to prevent an unnatural, abrupt mouth snap, but eliminate the 0.5s overshoot.
  const safeEndIndex = Math.min(pcmBytes.length, lastVoiceIndex + 3200);
  return pcmBytes.slice(0, safeEndIndex);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const Avatar3D = forwardRef((props, ref) => {
  const videoRef = useRef(null);
  const audioRef = useRef(null);
  const simliClientRef = useRef(null);
  const connectingRef = useRef(null);
  
  const [status, setStatus] = useState("connecting"); 
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);

  const closeStaleClient = () => {
    simliClientRef.current?.close?.();
    simliClientRef.current = null;
  };

  const connectFresh = async () => {
    setStatus("connecting");
    const res = await fetch(`${API_BASE}/simli-session`);
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
    ensureConnected();
    return () => closeStaleClient();
  }, []);

  useImperativeHandle(ref, () => ({
    speak: async (base64Audio) => {
      let client;
      try {
        client = await ensureConnected(); 
      } catch {
        return;
      }

      client.ClearBuffer?.();

      let pcm = wavBase64ToPcm16(base64Audio);
      
      // Apply the trimmer to chop off the TTS trailing static
      pcm = trimTrailingSilence(pcm);

      try {
        for (let offset = 0; offset < pcm.length; offset += CHUNK_SIZE) {
          client.sendAudioData(pcm.slice(offset, offset + CHUNK_SIZE));
          await sleep(2); 
        }
      } catch (err) {
        console.error("sendAudioData failed, discarding session:", err);
        closeStaleClient();
        setStatus("error");
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
            {status === "error" ? "Reconnecting…" : "Connecting to Receptionist…"}
          </div>
        </div>
      )}
    </div>
  );
});

export default Avatar3D;