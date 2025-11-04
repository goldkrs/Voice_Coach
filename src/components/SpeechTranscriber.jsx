import React, { useState, useRef } from "react";

const SpeechTranscriber = () => {
  const [transcript, setTranscript] = useState("");
  const [fillerCount, setFillerCount] = useState(0);
  const [silenceDuration, setSilenceDuration] = useState(0);
  const [sentiment, setSentiment] = useState("unknown");
  const [error, setError] = useState("");
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
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
        setTranscript(data.transcript || "No transcript received.");
        setFillerCount(data.filler_count || 0);
        setSilenceDuration(data.silence_duration || 0);
        setSentiment(data.sentiment || "unknown");
        setError(data.error || "");
      } catch (err) {
        console.error("Error contacting backend:", err);
        setTranscript("Error contacting backend.");
        setError(err.message);
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

      <div style={{ marginTop: "2rem" }}>
        <p><strong>Transcript:</strong> {transcript}</p>
        <p><strong>Filler words detected:</strong> {fillerCount}</p>
        <p><strong>Silence duration:</strong> {silenceDuration} seconds</p>
        <p><strong>Sentiment:</strong> {sentiment}</p>
        {error && <p style={{ color: "red" }}><strong>Error:</strong> {error}</p>}
      </div>
    </div>
  );
};

export default SpeechTranscriber;