"use client";
import { useState, useEffect } from "react";

export default function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("theme");
    const auto = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setDark(saved ? saved === "dark" : auto);
  }, []);

  const toggle = () => {
    const next = !dark;
    setDark(next);
    if (next) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  return (
    <button className="theme-toggle" onClick={toggle} aria-label="Toggle theme">
      {dark ? "☀" : "◑"}
    </button>
  );
}