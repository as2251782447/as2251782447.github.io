"use client";
import { useEffect, useRef } from "react";

export default function ParticlesDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    interface Particle {
      x: number; y: number; vx: number; vy: number;
      size: number; alpha: number; color: string;
    }

    const particles: Particle[] = Array.from({ length: 120 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.8,
      vy: (Math.random() - 0.5) * 0.8,
      size: Math.random() * 3 + 1,
      alpha: Math.random() * 0.5 + 0.3,
      color: ["#c96438", "#d4a574", "#e07a4f"][Math.floor(Math.random() * 3)],
    }));

    let mouse = { x: -999, y: -999 };

    const onMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };
    window.addEventListener("mousemove", onMouseMove);

    function dist(a: Particle, b: Particle) {
      return Math.hypot(a.x - b.x, a.y - b.y);
    }

    function draw() {
      ctx.fillStyle = "rgba(15, 14, 12, 0.1)";
      ctx.fillRect(0, 0, width, height);

      for (const p of particles) {
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const d = Math.hypot(dx, dy);
        if (d < 120) {
          const force = (120 - d) / 120 * 0.03;
          p.vx -= (dx / d) * force;
          p.vy -= (dy / d) * force;
        }

        p.x += p.vx;
        p.y += p.vy;
        p.vx *= 0.99;
        p.vy *= 0.99;

        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }

      // connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const d = dist(particles[i], particles[j]);
          if (d < 100) {
            ctx.globalAlpha = (1 - d / 100) * 0.15;
            ctx.strokeStyle = "#d4a574";
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      ctx.globalAlpha = 1;
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
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <main className="min-h-screen bg-[#0f0e0c] text-[#e8e0d8]">
      <div className="fixed top-4 left-4 z-50">
        <a href="/lab" className="text-xs opacity-60 hover:opacity-100 transition-opacity">← Lab</a>
        <p className="text-xs opacity-40 mt-1">move mouse to repel</p>
      </div>
      <canvas ref={canvasRef} className="w-full h-screen" style={{ display: "block" }} />
    </main>
  );
}