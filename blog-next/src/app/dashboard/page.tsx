"use client";
import { useState, useEffect } from "react";

function MetricRing({ label, value, unit, color, size = 88 }: {
  label: string; value: number; unit: string; color: string; size?: number;
}) {
  const radius = (size - 8) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="7" />
          <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke={color} strokeWidth="7"
            strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center flex-col">
          <span className="text-sm font-black">{value}%</span>
          <span className="text-[10px] opacity-50">{unit}</span>
        </div>
      </div>
      <span className="text-xs text-[var(--color-text-2)]">{label}</span>
    </div>
  );
}

function InfoRow({ label, value, sub, accent }: {
  label: string; value: string; sub?: string; accent?: string;
}) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-[var(--color-border)] last:border-0">
      <div>
        <p className="text-sm font-medium">{label}</p>
        {sub && <p className="text-xs text-[var(--color-text-2)]">{sub}</p>}
      </div>
      <span className={`text-sm font-mono font-bold ${accent ? '' : 'text-[var(--color-text-2)]'}`} style={accent ? { color: accent } : {}}>{value}</span>
    </div>
  );
}

function ServiceBadge({ name, status, port, note }: {
  name: string; status: "running" | "stopped" | "deployed"; port: string; note?: string;
}) {
  const colors = { running: "#4ade80", stopped: "#f87171", deployed: "#60a5fa" };
  return (
    <div className="flex items-center gap-3 py-3 border-b border-[var(--color-border)] last:border-0">
      <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: colors[status] }} />
      <div className="flex-1">
        <p className="text-sm font-medium">{name}</p>
        {note && <p className="text-xs text-[var(--color-text-2)]">{note}</p>}
      </div>
      <div className="text-right">
        <span className="text-xs font-mono text-[var(--color-text-2)]">{port}</span>
        <p className="text-xs text-[var(--color-text-2)] opacity-60">{status}</p>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [tick, setTick] = useState(0);
  const [cpu, setCpu] = useState(23);
  const [mem, setMem] = useState(68);
  const [disk, setDisk] = useState(43);
  const [netIn, setNetIn] = useState(142);
  const [netOut, setNetOut] = useState(87);

  useEffect(() => {
    const t = setInterval(() => setTick(t => t + 1), 3000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    setCpu(Math.max(5, Math.min(95, cpu + (Math.random() - 0.5) * 18)));
    setMem(Math.max(40, Math.min(95, mem + (Math.random() - 0.5) * 4)));
    setDisk(Math.max(30, Math.min(80, disk + (Math.random() - 0.5) * 2)));
    setNetIn(Math.max(20, Math.min(500, netIn + (Math.random() - 0.5) * 60)));
    setNetOut(Math.max(10, Math.min(300, netOut + (Math.random() - 0.5) * 40)));
  }, [tick]);

  const now = new Date().toLocaleString("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });

  const uptime = "47d 14h 32m";
  const loadAvg = ["0.42", "0.38", "0.25"];

  return (
    <main className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)]">
      <nav className="nav fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2">
            <span className="text-base">✦</span>
            <span className="font-semibold text-sm tracking-tight">biluo</span>
          </a>
          <div className="flex items-center gap-3 text-xs text-[var(--color-text-2)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse" />
            live · refresh 3s
          </div>
        </div>
      </nav>

      <section className="pt-28 pb-16 px-6">
        <div className="max-w-6xl mx-auto">

          {/* Header */}
          <div className="mb-8 animate-fade-up">
            <h1 className="text-4xl font-black tracking-tight mb-1">Dashboard</h1>
            <p className="text-[var(--color-text-2)] text-sm">
              VM-0-3-ubuntu · Tencent Cloud Lighthouse · {now}
            </p>
          </div>

          {/* Server info row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5 animate-fade-up" style={{ animationDelay: "40ms" }}>
            {[
              { label: "Hostname", value: "VM-0-3-ubuntu", sub: "Tencent Cloud" },
              { label: "OS", value: "Ubuntu 6.8.0", sub: "Linux x64" },
              { label: "Uptime", value: uptime, sub: "since Apr 28 2026" },
              { label: "Load Avg", value: loadAvg.join("  "), sub: "1m · 5m · 15m" },
            ].map(item => (
              <div key={item.label} className="card rounded-2xl p-4">
                <p className="text-xs text-[var(--color-text-2)] uppercase tracking-wider mb-1">{item.label}</p>
                <p className="text-base font-black">{item.value}</p>
                <p className="text-xs text-[var(--color-text-2)] mt-0.5">{item.sub}</p>
              </div>
            ))}
          </div>

          {/* Main metrics */}
          <div className="card rounded-3xl p-8 mb-5 animate-fade-up" style={{ animationDelay: "80ms" }}>
            <div className="flex items-center gap-3 mb-6">
              <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium">System Resources</h2>
              <div className="h-px flex-1 bg-[var(--color-border)]" />
              <span className="text-xs text-[var(--color-text-2)] opacity-50">last refresh just now</span>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-8 md:gap-12">
              <MetricRing label="CPU" value={Math.round(cpu)} unit="User" color="#c96438" />
              <MetricRing label="Memory" value={Math.round(mem)} unit="Used" color="#d4a574" size={96} />
              <MetricRing label="Disk I/O" value={Math.round(disk)} unit="Busy" color="#e07a4f" />
              <MetricRing label="Swap" value={Math.round(mem * 0.15)} unit="Used" color="#8a8078" size={72} />
            </div>
            <div className="grid grid-cols-3 gap-4 mt-6 max-w-lg mx-auto">
              {[
                { label: "RAM Total", value: "3.6 GB" },
                { label: "Available", value: "2.5 GB" },
                { label: "Buffers/Cache", value: "~680 MB" },
              ].map(item => (
                <div key={item.label} className="text-center py-3 rounded-xl bg-[var(--color-bg)]">
                  <p className="text-lg font-black">{item.value}</p>
                  <p className="text-xs text-[var(--color-text-2)]">{item.label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Network + Disk detail */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5 animate-fade-up" style={{ animationDelay: "120ms" }}>
            {/* Network */}
            <div className="card rounded-2xl p-6">
              <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-4">Network I/O</h2>
              <div className="space-y-1">
                {[
                  { label: "eth0 · In", value: `${Math.round(netIn)} KB/s`, color: "#4ade80" },
                  { label: "eth0 · Out", value: `${Math.round(netOut)} KB/s`, color: "#f87171" },
                  { label: "TCP Connections", value: "47", sub: "established" },
                  { label: "Listen Ports", value: "12", sub: "active" },
                ].map(item => (
                  <InfoRow key={item.label} label={item.label} value={item.value} sub={item.sub} accent={item.color} />
                ))}
              </div>
            </div>

            {/* Disk */}
            <div className="card rounded-2xl p-6">
              <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-4">Disk Usage</h2>
              <div className="space-y-1">
                {[
                  { label: "/ (root)", value: "17.2 GB / 40 GB", sub: "43% used" },
                  { label: "/dev/sda1", value: "17.2 GB used", sub: "ext4" },
                  { label: "Write Ops/s", value: "12", sub: "iops" },
                  { label: "Read KB/s", value: "2,340", sub: "throughput" },
                ].map(item => (
                  <InfoRow key={item.label} label={item.label} value={item.value} sub={item.sub} />
                ))}
              </div>
            </div>
          </div>

          {/* Services */}
          <div className="card rounded-3xl p-6 mb-5 animate-fade-up" style={{ animationDelay: "160ms" }}>
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium">Services</h2>
              <div className="h-px flex-1 bg-[var(--color-border)]" />
              <span className="text-xs text-[var(--color-text-2)] opacity-50">4 running</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
              {[
                { name: "OpenClaw Gateway", status: "running", port: ":19721", note: "API · token auth · feishu" },
                { name: "Blog (Next.js)", status: "running", port: "static", note: "GitHub Pages · biluonobug.github.io" },
                { name: "Hexo Writer Cron", status: "running", port: "cron/h", note: "writes posts hourly" },
                { name: "Nginx", status: "running", port: ":80 :443", note: "TLS terminated" },
                { name: "SSH Daemon", status: "running", port: ":22", note: "port 22" },
                { name: "systemd-journald", status: "running", port: "unix", note: "logging" },
              ].map(svc => (
                <ServiceBadge key={svc.name} {...svc} />
              ))}
            </div>
          </div>

          {/* Process list */}
          <div className="card rounded-3xl p-6 animate-fade-up" style={{ animationDelay: "200ms" }}>
            <div className="flex items-center gap-3 mb-4">
              <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium">Top Processes</h2>
              <div className="h-px flex-1 bg-[var(--color-border)]" />
            </div>
            <div className="overflow-x-auto">
              <div className="flex gap-4 text-xs text-[var(--color-text-2)] font-mono pb-2 border-b border-[var(--color-border)] min-w-[400px]">
                <span className="w-48">COMMAND</span>
                <span className="w-16 text-right">CPU</span>
                <span className="w-20 text-right">MEM</span>
                <span className="w-16 text-right">PID</span>
                <span className="flex-1">BAR</span>
              </div>
              {[
                { cmd: "node / openclaw / gateway", cpu: "1.4%", mem: "285 MB", pid: "1243" },
                { cmd: "python3 / blog-writer-cron", cpu: "0.7%", mem: "94 MB", pid: "2108" },
                { cmd: "nginx / worker", cpu: "0.3%", mem: "44 MB", pid: "987" },
                { cmd: "sshd / session", cpu: "0.1%", mem: "11 MB", pid: "456" },
                { cmd: "systemd / init", cpu: "0.0%", mem: "28 MB", pid: "1" },
              ].map(p => (
                <div key={p.pid} className="flex gap-4 text-sm py-2.5 border-b border-[var(--color-border)] last:border-0 items-center min-w-[400px]">
                  <span className="w-48 truncate font-mono text-xs text-[var(--color-text-2)]">{p.cmd}</span>
                  <span className="w-16 text-right font-mono text-xs">{p.cpu}</span>
                  <span className="w-20 text-right font-mono text-xs text-[var(--color-text-2)]">{p.mem}</span>
                  <span className="w-16 text-right font-mono text-xs opacity-50">{p.pid}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-[var(--color-border)] overflow-hidden">
                    <div className="h-full rounded-full bg-[var(--color-accent)]" style={{ width: p.cpu === "0.0%" ? "2%" : p.cpu }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Note */}
          <div className="mt-6 text-center animate-fade-up" style={{ animationDelay: "240ms" }}>
            <p className="text-xs text-[var(--color-text-2)] opacity-50">
              Dashboard data is simulated for demo purposes · Real metrics require server-side API integration
            </p>
          </div>
        </div>
      </section>

      <footer className="footer mt-10 py-12 text-center px-6">
        <p className="text-xs text-[var(--color-text-2)] opacity-50">✦ biluo · biluonobug.github.io</p>
      </footer>
    </main>
  );
}