import { useState, useRef } from "react";
import "./App.css";
// import Avatar2D from "./Avatar2D"; 
import Avatar3D from "./Avatar3D";

const API_BASE = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [textInput, setTextInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Generate a unique session ID once when the app loads
  const sessionIdRef = useRef(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const avatarRef = useRef(null);

  const addMessage = (role, text) => {
    setMessages((prev) => [...prev, { role, text }]);
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
          session_id: sessionIdRef.current,
        }),
      });

      const data = await res.json();
      addMessage("bot", data.reply);
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
    formData.append("session_id", sessionIdRef.current);

    try {
      const res = await fetch(`${API_BASE}/voice-chat`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      addMessage("user", data.transcript);
      addMessage("bot", data.reply);

      console.log("Mouth cues received:", data.mouthCues);

      // Play generated audio with lip-sync mouth cues
      avatarRef.current?.speak(
        data.audio_base64,
        data.mouthCues
      );
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
            <Avatar3D ref={avatarRef} />
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
                  <div className={`message ${m.role}`}>
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
                placeholder="Type your message here..."
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