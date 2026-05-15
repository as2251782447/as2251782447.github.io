"use client";
import { useState, useEffect } from "react";

const phrases = [
  "console.log('hello world');",
  "const agent = await llm.complete(task);",
  "SELECT * FROM memory WHERE relevance > 0.8;",
  "git commit -m 'fix: finally works'",
  "docker compose up -d --build",
  "await agent.remember(everything);",
  "npm install --save-dev openclaw",
  "curl -X POST https://api.example.com",
  "SELECT vector FROM embed WHERE id = $1;",
  "systemctl restart openclaw",
];

export default function TypingDemo() {
  const [text, setText] = useState("");
  const [phraseIdx, setPhraseIdx] = useState(0);
  const [charIdx, setCharIdx] = useState(0);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const phrase = phrases[phraseIdx];
    const speed = deleting ? 30 : 80;

    const timeout = setTimeout(() => {
      if (!deleting) {
        if (charIdx < phrase.length) {
          setText(phrase.slice(0, charIdx + 1));
          setCharIdx(c => c + 1);
        } else {
          setTimeout(() => setDeleting(true), 1500);
        }
      } else {
        if (charIdx > 0) {
          setText(phrase.slice(0, charIdx - 1));
          setCharIdx(c => c - 1);
        } else {
          setDeleting(false);
          setPhraseIdx(i => (i + 1) % phrases.length);
        }
      }
    }, speed);

    return () => clearTimeout(timeout);
  }, [charIdx, deleting, phraseIdx]);

  return (
    <main className="min-h-screen bg-[#0f0e0c] flex items-center justify-center px-6">
      <div className="fixed top-4 left-4 z-50">
        <a href="/lab" className="text-xs opacity-60 hover:opacity-100 transition-opacity">← Lab</a>
      </div>
      <div className="text-center">
        <div className="font-mono text-lg md:text-2xl text-[#e07a4f] min-h-[2em] flex items-center justify-center gap-1">
          <span>{text}</span>
          <span className="w-0.5 h-5 bg-[#e07a4f] animate-pulse inline-block ml-0.5" />
        </div>
        <p className="mt-6 text-xs text-[#6b6560]">typewriter effect · cursor blink</p>
      </div>
    </main>
  );
}