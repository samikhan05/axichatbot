import { useState, useRef } from "react";
import "./App.css";
import AvatarSimli from "./AvatarSimli";

const API_BASE = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [textInput, setTextInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Generate a unique session ID once when the app loads
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const avatarRef = useRef(null);
  const speechAbortRef = useRef(null);

  const addMessage = (role, text, extra = {}) => {
    setMessages((prev) => [...prev, { role, text, ...extra }]);
  };

  // Ask the backend for speech + visemes so the avatar speaks typed replies too
  const speakText = async (text, language) => {
    // a new reply interrupts the previous one
    speechAbortRef.current?.abort();
    const controller = new AbortController();
    speechAbortRef.current = controller;
    try {
      const res = await fetch(`${API_BASE}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) return;
      await avatarRef.current?.speakStream(res, controller.signal);
    } catch (err) {
      if (err.name !== "AbortError") console.warn("TTS unavailable:", err);
    }
  };

  // ---- Text chat ----
  const sendTextMessage = async () => {
    if (!textInput.trim()) return;

    const question = textInput;
    setTextInput("");
    addMessage("user", question);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: question,
          session_id: sessionId,
        }),
      });

      const data = await res.json();
      addMessage("bot", data.reply, { sources: data.sources, language: data.language });
      speakText(data.reply, data.language); // text shows now, voice follows
    } catch (err) {
      console.error("Text chat failed:", err);
      addMessage(
        "bot",
        "Sorry, something went wrong connecting to the server."
      );
    }

    setIsLoading(false);
  };

  // ---- Voice chat ----
  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    const recorder = new MediaRecorder(stream, {
      mimeType: "audio/webm",
    });

    audioChunksRef.current = [];

    recorder.ondataavailable = (e) => {
      audioChunksRef.current.push(e.data);
    };

    recorder.onstop = handleRecordingStop;

    mediaRecorderRef.current = recorder;
    recorder.start();

    setIsRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  const handleRecordingStop = async () => {
    setIsLoading(true);

    const audioBlob = new Blob(audioChunksRef.current, {
      type: "audio/webm",
    });

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("session_id", sessionId);

    try {
      const res = await fetch(`${API_BASE}/voice-chat`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.transcript) addMessage("user", data.transcript);
      addMessage("bot", data.reply, { sources: data.sources, language: data.language });

      // text is shown now; speech streams in right behind it
      speakText(data.reply, data.language);
    } catch (err) {
      console.error("Voice chat failed:", err);

      addMessage(
        "bot",
        "Sorry, something went wrong processing your voice message."
      );
    }

    setIsLoading(false);
  };

  return (
    <div className="app-wrapper">
      <div className="app-container">
        
        {/* UPDATED HEADER SECTION */}
        <header className="app-header">
          <div className="header-logo">
            {/* Make sure your logo is in the public folder or update the path */}
            <img src="/logo.png" alt="Company Logo" />
          </div>
          <div className="header-titles">
            <h1>Axichatbot Receptionist</h1>
            <p>Your Virtual Assistant</p>
          </div>
        </header>

        <div className="app-body">
          {/* Avatar Section */}
          <div className="avatar-section">
            <AvatarSimli ref={avatarRef} />
          </div>

          {/* Chat Section */}
          <div className="chat-section">
            <div className="chat-window">
              {messages.length === 0 && (
                <div className="welcome-message">
                  Hi there! How can I help you today?
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`message-wrapper ${m.role}`}>
                  {/* dir="auto" renders Urdu script right-to-left */}
                  <div className={`message ${m.role}`} dir="auto">
                    {m.text}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="message-wrapper bot">
                  <div className="message bot loading">
                    <span className="dot"></span>
                    <span className="dot"></span>
                    <span className="dot"></span>
                  </div>
                </div>
              )}
            </div>

            <div className="controls">
              <input
                type="text"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyDown={(e) =>
                  e.key === "Enter" && sendTextMessage()
                }
                dir="auto"
                placeholder="Type in English, Urdu or Roman Urdu..."
              />

              <button
                onClick={sendTextMessage}
                disabled={isLoading || !textInput.trim()}
                className="btn-send"
              >
                Send
              </button>

              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`btn-voice ${isRecording ? "recording" : ""}`}
                disabled={isLoading}
              >
                {isRecording ? "Stop" : "Speak"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;