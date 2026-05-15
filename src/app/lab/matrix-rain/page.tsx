"use client";
import { useEffect, useRef } from "react";

export default function MatrixRainDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const chars = "日木水火土金电人工智能神经网络机器学习深度学习算法架构系统设计前端后端数据库云原生容器编排";
    const fontSize = 14;
    const columns = Math.floor(canvas.width / fontSize);
    const drops: number[] = Array(columns).fill(1);
    const colors = ["#c96438", "#d4a574", "#e07a4f", "#8a8078"];

    function draw() {
      ctx.fillStyle = "rgba(15, 14, 12, 0.05)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.font = `${fontSize}px monospace`;

      for (let i = 0; i < drops.length; i++) {
        const char = chars[Math.floor(Math.random() * chars.length)];
        const color = colors[Math.floor(Math.random() * colors.length)];
        ctx.fillStyle = color;
        ctx.fillText(char, i * fontSize, drops[i] * fontSize);

        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    }

    const interval = setInterval(draw, 50);
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", resize);
    return () => {
      clearInterval(interval);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <main className="min-h-screen bg-[#0f0e0c] text-[#e8e0d8]">
      <div className="fixed top-4 left-4 z-50">
        <a href="/lab" className="text-xs opacity-60 hover:opacity-100 transition-opacity">
          ← Lab
        </a>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full h-screen"
        style={{ display: "block" }}
      />
    </main>
  );
}