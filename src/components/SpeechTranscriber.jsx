// frontend/src/components/SpeechTranscriber.jsx
import React, { useEffect, useState, useRef, useMemo } from "react";
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

const normSentiment = (s) => {
  if (s === null || s === undefined) return 0;
  const t = String(s).trim().toLowerCase();
  if (t.startsWith("pos")) return 1;
  if (t.startsWith("neg")) return -1;
  if (t === "neutral") return 0;
  return 0;
};

export default function SpeechTranscriber() {
  const [transcript, setTranscript] = useState("");
  const [fillerCount, setFillerCount] = useState(0);
  const [silenceDuration, setSilenceDuration] = useState(0);
  const [sentiment, setSentiment] = useState("unknown");
  const [segments, setSegments] = useState([]);
  const [error, setError] = useState("");
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [savedId, setSavedId] = useState(null);
  const [loggedIn, setLoggedIn] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("http://localhost:8000/me", { credentials: "include" });
        setLoggedIn(res.ok);
      } catch {
        setLoggedIn(false);
      }
    })();
  }, []);

  useEffect(() => {
    console.log("segments:", segments);
  }, [segments]);

  const startRecording = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);

      mediaRecorder.onstop = async () => {
        setLoading(true);
        try {
          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
          const formData = new FormData();
          formData.append("file", audioBlob, "speech.webm");

          const res = await fetch("http://localhost:8000/transcribe", {
            method: "POST",
            body: formData,
            credentials: "include",
          });

          const data = await res.json().catch(() => ({ error: "Invalid JSON response" }));
          if (!res.ok) {
            setError(data.detail || data.error || "Transcription failed");
            setTranscript("");
            setSegments([]);
          } else {
            setTranscript(data.transcript || "");
            setFillerCount(data.filler_count || 0);
            setSilenceDuration(data.silence_duration || 0);
            setSentiment(data.sentiment || "unknown");
            setSegments(data.sentiment_segments || []);
            setSavedId(data.saved_id || null);
            setError(data.error || "");
          }
        } catch (err) {
          console.error("Error contacting backend:", err);
          setError(err.message || "Network error");
        } finally {
          setLoading(false);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorderRef.current.start();
      setRecording(true);
    } catch (err) {
      console.error("Microphone access error:", err);
      setError("Unable to access microphone: " + (err.message || err));
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      try {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current.stream?.getTracks().forEach((t) => t.stop());
      } catch (e) {
        // ignore
      }
      setRecording(false);
    }
  };

  const chartData = useMemo(() => {
    const labels = segments.map((s) => `${s.start}s`);
    const data = segments.map((s) => normSentiment(s.sentiment));
    return {
      labels,
      datasets: [
        {
          label: "Sentiment over time (1 pos, 0 neutral, -1 neg)",
          data,
          borderColor: "rgba(75,192,192,1)",
          backgroundColor: "rgba(75,192,192,0.2)",
          tension: 0.3,
          pointRadius: 6,
          borderWidth: 2,
          fill: false,
        },
      ],
    };
  }, [segments]);

  const options = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: "category" },
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
  }, [segments]);

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h2>Public Speaking Coach</h2>

      <div style={{ marginBottom: 12 }}>
        <button onClick={startRecording} disabled={recording || loading} style={{ marginRight: 8 }}>
          {recording ? "Recording…" : "Start"}
        </button>
        <button onClick={stopRecording} disabled={!recording || loading}>
          Stop
        </button>
        <span style={{ marginLeft: 12 }}>{loading ? "Analyzing…" : loggedIn ? "Logged in" : "Not logged in"}</span>
      </div>

      <div style={{ marginTop: "1rem" }}>
        <p>
          <strong>Transcript:</strong> {transcript || "—"}
        </p>
        <p>
          <strong>Filler words detected:</strong> {fillerCount}
        </p>
        <p>
          <strong>Silence duration:</strong> {silenceDuration} seconds
        </p>
        <p>
          <strong>Overall Sentiment:</strong> {sentiment}
        </p>
        {savedId && (
          <p>
            <strong>Saved ID:</strong> {savedId}
          </p>
        )}
      </div>

      <div style={{ marginTop: "2rem", maxWidth: "800px" }}>
        <h4>Sentiment Timeline</h4>
        {segments.length > 0 ? (
          <div style={{ height: 300 }}>
            <Line key={segments.length} data={chartData} options={options} />
          </div>
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
                <strong>
                  {s.start}s - {s.end}s:
                </strong>{" "}
                {s.sentiment} — {s.transcript}
              </li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <p style={{ color: "red" }}>
          <strong>Error:</strong> {error}
        </p>
      )}
    </div>
  );
}