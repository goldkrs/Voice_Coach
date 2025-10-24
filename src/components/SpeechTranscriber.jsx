import React, { useState, useRef } from "react";
import { encodeWAV, resampleBuffer } from "../utils/wavEncoder";

const SpeechTranscriber = () => {
  const [transcript, setTranscript] = useState("");
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const audioBufferRef = useRef([]);

  const startRecording = async () => {
    audioBufferRef.current = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContextRef.current.createMediaStreamSource(stream);

    const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      audioBufferRef.current.push(new Float32Array(input));
    };

    source.connect(processor);
    processor.connect(audioContextRef.current.destination);
    scriptProcessorRef.current = processor;
  };

  const stopRecording = async () => {
    scriptProcessorRef.current.disconnect();
    mediaStreamRef.current.getTracks().forEach((track) => track.stop());

    const flatBuffer = Float32Array.from(audioBufferRef.current.flat());
    const resampled = resampleBuffer(flatBuffer, audioContextRef.current.sampleRate, 16000);
    const wavBlob = encodeWAV(resampled, 16000);

    const formData = new FormData();
    formData.append("file", wavBlob, "speech.wav");

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