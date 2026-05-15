"use client";
import { useEffect, useRef } from "react";

export default function StarfieldDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    interface Star {
      x: number; y: number; z: number;
      prevX: number; prevY: number;
    }

    const stars: Star[] = Array.from({ length: 400 }, () => ({
      x: Math.random() * width - width / 2,
      y: Math.random() * height - height / 2,
      z: Math.random() * 1000,
      prevX: 0, prevY: 0,
    }));

    let speed = 2;
    const speedEl = document.getElementById("speed-label");

    function draw() {
      ctx.fillStyle = "rgba(15, 14, 12, 0.18)";
      ctx.fillRect(0, 0, width, height);

      for (const star of stars) {
        star.prevX = (star.x / star.z) * 400 + width / 2;
        star.prevY = (star.y / star.z) * 400 + height / 2;
        star.z -= speed;
        if (star.z <= 0) {
          star.x = Math.random() * width - width / 2;
          star.y = Math.random() * height - height / 2;
          star.z = 1000;
        }

        const px = (star.x / star.z) * 400 + width / 2;
        const py = (star.y / star.z) * 400 + height / 2;
        const brightness = Math.max(0.1, 1 - star.z / 1000);
        ctx.strokeStyle = `rgba(224, 122, 79, ${brightness})`;
        ctx.lineWidth = brightness * 2;
        ctx.beginPath();
        ctx.moveTo(star.prevX, star.prevY);
        ctx.lineTo(px, py);
        ctx.stroke();
      }

      requestAnimationFrame(draw);
    }

    const interval = setInterval(() => {
      speed = Math.min(20, speed + 0.5);
      if (speedEl) speedEl.textContent = `${Math.round(speed)}x`;
    }, 2000);

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };
    window.addEventListener("resize", resize);

    draw();
    return () => {
      clearInterval(interval);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <main className="min-h-screen bg-[#0f0e0c] text-[#e8e0d8]">
      <div className="fixed top-4 left-4 z-50 flex flex-col gap-2">
        <a href="/lab" className="text-xs opacity-60 hover:opacity-100 transition-opacity">← Lab</a>
        <span id="speed-label" className="text-xs font-mono opacity-50">2x</span>
      </div>
      <canvas ref={canvasRef} className="w-full h-screen" style={{ display: "block" }} />
    </main>
  );
}