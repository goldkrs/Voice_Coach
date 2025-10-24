import React, { useState, useRef } from "react";

const SpeechTranscriber = () => {
  const [transcript, setTranscript] = useState("");
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream); // Chrome uses WebM
    audioChunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      audioChunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("file", audioBlob, "speech.webm");

      try {
        const res = await fetch("http://localhost:8000/transcribe", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        setTranscript(data.transcript || data.error || "No transcript received.");
      } catch (err) {
        console.error("Error sending to backend:", err);
        setTranscript("Error contacting backend.");
      }
    };

    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h2>🎙️ Public Speaking Coach</h2>
      <button onClick={startRecording} style={{ marginRight: "1rem" }}>
        Start
      </button>
      <button onClick={stopRecording}>Stop</button>
      <div
        style={{
          marginTop: "2rem",
          padding: "1rem",
          background: "#fff",
          border: "1px solid #ccc",
          minHeight: "100px",
        }}
      >
        {transcript}
      </div>
    </div>
  );
};

export default SpeechTranscriber;