import React, { useState, useRef } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  CategoryScale,
  Tooltip,
} from "chart.js";

ChartJS.register(LineElement, PointElement, LinearScale, Title, CategoryScale, Tooltip);

const mapSentimentToValue = (s) => {
  if (!s) return 0;
  s = s.toLowerCase();
  if (s.startsWith("pos")) return 1;
  if (s.startsWith("neg")) return -1;
  if (s === "neutral") return 0;
  return 0;
};

const SpeechTranscriber = () => {
  const [transcript, setTranscript] = useState("");
  const [fillerCount, setFillerCount] = useState(0);
  const [silenceDuration, setSilenceDuration] = useState(0);
  const [sentiment, setSentiment] = useState("unknown");
  const [segments, setSegments] = useState([]);
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
        setSegments(data.sentiment_segments || []);
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

  const labels = segments.map((s) => `${s.start}s`);
  const dataValues = segments.map((s) => mapSentimentToValue(s.sentiment));
  const chartData = {
    labels,
    datasets: [
      {
        label: "Sentiment over time (1 pos, 0 neutral, -1 neg)",
        data: dataValues,
        borderColor: "rgba(75,192,192,1)",
        backgroundColor: "rgba(75,192,192,0.2)",
        tension: 0.3,
        pointRadius: 6,
      },
    ],
  };

  const options = {
    scales: {
      y: {
        min: -1.2,
        max: 1.2,
        ticks: {
          stepSize: 1,
          callback: function (value) {
            if (value === 1) return "Positive";
            if (value === 0) return "Neutral";
            if (value === -1) return "Negative";
            return value;
          },
        },
      },
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: function (context) {
            const idx = context.dataIndex;
            const seg = segments[idx];
            return seg ? `${seg.sentiment} — ${seg.transcript}` : "";
          },
        },
      },
    },
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h2>🎙️ Public Speaking Coach</h2>
      <button onClick={startRecording} style={{ marginRight: "1rem" }}>
        Start
      </button>
      <button onClick={stopRecording}>Stop</button>

      <div style={{ marginTop: "1.5rem" }}>
        <p><strong>Transcript:</strong> {transcript}</p>
        <p><strong>Filler words detected:</strong> {fillerCount}</p>
        <p><strong>Silence duration:</strong> {silenceDuration} seconds</p>
        <p><strong>Overall Sentiment:</strong> {sentiment}</p>
      </div>

      <div style={{ marginTop: "2rem", maxWidth: "800px" }}>
        <h4>Sentiment Timeline</h4>
        {segments.length > 0 ? (
          <Line data={chartData} options={options} />
        ) : (
          <p>No sentiment segments yet. Record something and stop to analyze.</p>
        )}
      </div>

      {segments.length > 0 && (
        <div style={{ marginTop: "1rem" }}>
          <h4>Segments</h4>
          <ul>
            {segments.map((s, idx) => (
              <li key={idx}>
                <strong>{s.start}s - {s.end}s:</strong> {s.sentiment} — {s.transcript}
              </li>
            ))}
          </ul>
        </div>
      )}

      {error && <p style={{ color: "red" }}><strong>Error:</strong> {error}</p>}
    </div>
  );
};

export default SpeechTranscriber;