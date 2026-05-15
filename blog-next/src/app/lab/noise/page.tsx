"use client";
import { useEffect, useRef } from "react";

export default function NoiseFlowDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    const SCALE = 20;
    const cols = Math.floor(width / SCALE);
    const rows = Math.floor(height / SCALE);

    // Simple noise approximation using sine waves
    let t = 0;

    function noise(x: number, y: number, z: number): number {
      return (
        Math.sin(x * 0.1 + z * 0.5) * 0.5 +
        Math.sin(y * 0.15 + z * 0.3) * 0.3 +
        Math.sin((x + y) * 0.08 + z * 0.4) * 0.2
      );
    }

    function draw() {
      ctx.fillStyle = "rgba(15, 14, 12, 0.15)";
      ctx.fillRect(0, 0, width, height);

      t += 0.02;

      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const angle = (noise(i * 0.3, j * 0.3, t) + 1) * Math.PI;

          const cx = i * SCALE + SCALE / 2;
          const cy = j * SCALE + SCALE / 2;
          const len = SCALE * 0.4;

          const ex = cx + Math.cos(angle) * len;
          const ey = cy + Math.sin(angle) * len;

          const hue = ((angle / (2 * Math.PI)) * 30 + 20);
          ctx.strokeStyle = `hsla(${hue + 25}, 70%, ${50 + noise(i, j, t) * 20}%, 0.6)`;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(ex, ey);
          ctx.stroke();
        }
      }

      requestAnimationFrame(draw);
    }

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };
    window.addEventListener("resize", resize);
    draw();
    return () => window.removeEventListener("resize", resize);
  }, []);

  return (
    <main className="min-h-screen bg-[#0f0e0c] text-[#e8e0d8]">
      <div className="fixed top-4 left-4 z-50">
        <a href="/lab" className="text-xs opacity-60 hover:opacity-100 transition-opacity">← Lab</a>
      </div>
      <canvas ref={canvasRef} className="w-full h-screen" style={{ display: "block" }} />
    </main>
  );
}