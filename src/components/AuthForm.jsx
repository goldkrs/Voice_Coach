// frontend/src/components/AuthForm.jsx
import React, { useState } from "react";

export default function AuthForm({ onAuth }) {
  const [mode, setMode] = useState("login"); // "login" or "register"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  async function submit(e) {
    e.preventDefault();
    setMsg("");
    setLoading(true);

    try {
      if (mode === "register") {
        const res = await fetch("http://localhost:8000/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ username: username.trim(), password }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          setMsg(data.detail || data.error || "Register failed");
          setLoading(false);
          return;
        }
        setMsg("Registered and logged in");
        onAuth && onAuth(true);
      } else {
        const res = await fetch("http://localhost:8000/login", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          credentials: "include",
          body: new URLSearchParams({ username: username.trim(), password }).toString(),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          setMsg(data.detail || data.error || "Login failed");
          setLoading(false);
          return;
        }
        setMsg("Logged in");
        onAuth && onAuth(true);
      }
    } catch (err) {
      console.error("Auth error:", err);
      setMsg("Network error");
    } finally {
      setLoading(false);
      setPassword("");
    }
  }

  return (
    <form onSubmit={submit} style={{ maxWidth: 420, padding: 12, fontFamily: "sans-serif" }}>
      <h3>{mode === "login" ? "Login" : "Register"}</h3>

      <label style={{ display: "block", marginBottom: 6 }}>
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="username"
          required
          minLength={3}
          style={{ width: "100%", padding: 8, boxSizing: "border-box" }}
        />
      </label>

      <label style={{ display: "block", marginBottom: 8 }}>
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="password"
          type="password"
          required
          minLength={6}
          style={{ width: "100%", padding: 8, boxSizing: "border-box" }}
        />
      </label>

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <button type="submit" disabled={loading} style={{ padding: "8px 12px" }}>
          {loading ? (mode === "login" ? "Logging in…" : "Registering…") : mode === "login" ? "Login" : "Register"}
        </button>

        <button
          type="button"
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setMsg("");
          }}
          style={{ padding: "8px 12px" }}
        >
          {mode === "login" ? "Switch to register" : "Switch to login"}
        </button>
      </div>

      {msg && <div style={{ marginTop: 8, color: "crimson" }}>{msg}</div>}
    </form>
  );
}