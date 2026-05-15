"use client";
import { useEffect, useRef } from "react";

export default function GradientBgDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    let t = 0;

    function draw() {
      t += 0.003;

      const grad = ctx.createLinearGradient(
        Math.sin(t) * width,
        Math.cos(t * 0.7) * height,
        width - Math.sin(t * 1.1) * width,
        height - Math.cos(t * 0.5) * height
      );

      const hue1 = (t * 20) % 360;
      const hue2 = (hue1 + 60) % 360;
      const hue3 = (hue2 + 120) % 360;

      grad.addColorStop(0, `hsla(${hue1}, 65%, 60%, 0.85)`);
      grad.addColorStop(0.5, `hsla(${hue2}, 70%, 55%, 0.8)`);
      grad.addColorStop(1, `hsla(${hue3}, 60%, 50%, 0.75)`);

      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

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
    <main className="min-h-screen">
      <div className="fixed top-4 left-4 z-50">
        <a href="/lab" className="text-xs opacity-60 hover:opacity-100 transition-opacity text-white">← Lab</a>
      </div>
      <canvas ref={canvasRef} className="w-full h-screen" style={{ display: "block" }} />
    </main>
  );
}