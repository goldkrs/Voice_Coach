// frontend/src/App.jsx
import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AuthForm from "./components/AuthForm";
import SpeechTranscriber from "./components/SpeechTranscriber";

function ProtectedRoute({ children }) {
  return children;
}

export default function App() {
  const [authed, setAuthed] = useState(null); // null = unknown
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("http://localhost:8000/me", { credentials: "include" });
        setAuthed(res.ok);
      } catch {
        setAuthed(false);
      } finally {
        setChecking(false);
      }
    })();
  }, []);

  if (checking) return <div style={{ padding: 24 }}>Checking session…</div>;

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={
            authed ? (
              <Navigate to="/app" replace />
            ) : (
              <AuthForm
                onAuth={(ok) => {
                  if (ok) setAuthed(true);
                }}
              />
            )
          }
        />
        <Route
          path="/app"
          element={
            authed ? (
              <ProtectedRoute>
                <MainApp onLogout={() => setAuthed(false)} />
              </ProtectedRoute>
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />
        <Route path="*" element={<Navigate to={authed ? "/app" : "/login"} replace />} />
      </Routes>
    </BrowserRouter>
  );
}

function MainApp({ onLogout }) {
  const doLogout = async () => {
    try {
      await fetch("http://localhost:8000/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch (e) {
      // ignore
    } finally {
      onLogout && onLogout();
    }
  };

  return (
    <div style={{ padding: 16 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1 style={{ margin: 0 }}>Public Speaking Coach</h1>
        <div>
          <button onClick={doLogout} style={{ padding: "6px 10px" }}>
            Logout
          </button>
        </div>
      </header>

      <main style={{ marginTop: 18 }}>
        <SpeechTranscriber />
      </main>
    </div>
  );
}