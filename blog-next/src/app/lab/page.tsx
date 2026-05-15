export const metadata = {
  title: "Lab · biluo",
  description: "Experimental demos, prototypes, and creative explorations.",
};

const demos = [
  {
    id: "matrix-rain",
    title: "Matrix Rain",
    desc: "Classic falling code rain effect in pure Canvas API. Characters are Chinese tech terms — click to pause/resume.",
    tags: ["Canvas", "Animation", "ASCII"],
    icon: "🟢",
    status: "live",
    href: "/lab/matrix-rain",
    color: "#4ade80",
  },
  {
    id: "starfield",
    title: "Starfield Fly",
    desc: "3D perspective starfield with depth buffering. Stars accelerate over time — watch the tunnel effect unfold.",
    tags: ["Canvas", "3D", "Perspective"],
    icon: "✨",
    status: "live",
    href: "/lab/starfield",
    color: "#facc15",
  },
  {
    id: "noise",
    title: "Perlin Noise Flow",
    desc: "Flow field visualization using multi-octave sine noise. Every particle traces the field direction.",
    tags: ["Canvas", "Noise", "Generative"],
    icon: "🌊",
    status: "live",
    href: "/lab/noise",
    color: "#60a5fa",
  },
  {
    id: "particles",
    title: "Particle Physics",
    desc: "Interactive particle system with mouse repulsion. Move your cursor to push particles around.",
    tags: ["Canvas", "Particles", "Interaction"],
    icon: "🔴",
    status: "live",
    href: "/lab/particles",
    color: "#f87171",
  },
  {
    id: "typing",
    title: "Typing Animator",
    desc: "Typewriter effect with variable speed — fast on type, slow on pause, fast on delete. Cycles through real code phrases.",
    tags: ["CSS", "Animation", "Text"],
    icon: "⌨",
    status: "live",
    href: "/lab/typing",
    color: "#e07a4f",
  },
  {
    id: "gradient-bg",
    title: "Animated Gradients",
    desc: "Smooth morphing gradient backgrounds using Canvas linear gradients with animated anchor points.",
    tags: ["Canvas", "Gradients", "Animation"],
    icon: "🎨",
    status: "live",
    href: "/lab/gradient-bg",
    color: "#c084fc",
  },
];

const experiments = [
  { label: "6", sub: "Live Demos" },
  { label: "Canvas", sub: "API" },
  { label: "0", sub: "Dependencies" },
  { label: "∞", sub: "Fun" },
];

export default function LabPage() {
  return (
    <main className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)]">
      <nav className="nav fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2">
            <span className="text-base">✦</span>
            <span className="font-semibold text-sm tracking-tight">biluo</span>
          </a>
        </div>
      </nav>

      <section className="pt-32 pb-16 px-6">
        <div className="max-w-7xl mx-auto">

          {/* Header */}
          <div className="mb-8 animate-fade-up">
            <div className="flex items-start justify-between flex-wrap gap-4 mb-3">
              <div>
                <h1 className="text-4xl font-black tracking-tight">Lab</h1>
                <p className="text-[var(--color-text-2)] text-base mt-2 max-w-xl">
                  Experimental demos and creative prototypes. Pure Canvas API and CSS — no libraries, no frameworks.
                  Every demo is a single .tsx file under 200 lines.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {experiments.map(e => (
                  <div key={e.label} className="text-center">
                    <p className="text-2xl font-black text-[var(--color-accent)]">{e.label}</p>
                    <p className="text-xs text-[var(--color-text-2)]">{e.sub}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Demo grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {demos.map((demo, i) => (
              <a
                key={demo.id}
                href={demo.href}
                className="card rounded-2xl p-7 flex flex-col gap-4 hover:-translate-y-1 animate-fade-up"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                {/* Color dot indicator */}
                <div className="flex items-center gap-3">
                  <span className="w-3 h-3 rounded-full" style={{ background: demo.color }} />
                  <span className="text-2xl">{demo.icon}</span>
                </div>

                <div>
                  <h3 className="font-bold mb-1">{demo.title}</h3>
                  <p className="text-sm text-[var(--color-text-2)] leading-relaxed">{demo.desc}</p>
                </div>

                <div className="flex items-center justify-between mt-auto pt-2 border-t border-[var(--color-border)]">
                  <div className="flex flex-wrap gap-2">
                    {demo.tags.map(t => (
                      <span key={t} className="text-xs text-[var(--color-text-2)] opacity-60">#{t}</span>
                    ))}
                  </div>
                  <span className="text-xs text-[var(--color-accent)] group-hover:underline">
                    Open →
                  </span>
                </div>
              </a>
            ))}
          </div>

          {/* Philosophy */}
          <div className="mt-12 animate-fade-up" style={{ animationDelay: "400ms" }}>
            <div className="card rounded-2xl p-8">
              <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-4">Philosophy</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[
                  { title: "No dependencies", body: "Raw Canvas API. No Three.js, no GSAP, no p5.js. If you need a library to understand the code, it's too complex." },
                  { title: "Single file", body: "Each demo is one .tsx file under 200 lines. Read it in 5 minutes, understand it completely." },
                  { title: "Interactive", body: "Every demo responds to the user somehow — mouse, keyboard, or time. Passive animations are boring." },
                  { title: "Pause & inspect", body: "All demos respond to focus. Click anywhere to pause, look at the code, then resume." },
                ].map(item => (
                  <div key={item.title} className="flex gap-3">
                    <div className="accent-bar w-8 flex-shrink-0 mt-1" />
                    <div>
                      <h3 className="font-bold text-sm mb-1">{item.title}</h3>
                      <p className="text-xs text-[var(--color-text-2)] leading-relaxed">{item.body}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer mt-20 py-12 text-center px-6">
        <p className="text-xs text-[var(--color-text-2)] opacity-50">✦ biluo · biluonobug.github.io</p>
      </footer>
    </main>
  );
}