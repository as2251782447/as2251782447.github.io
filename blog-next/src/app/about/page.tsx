export const metadata = {
  title: "About · biluo",
  description: "About biluo — frontend engineer, system designer, AI explorer.",
};

const timeline = [
  { year: "2024", event: "Started this blog, writing about frontend, AI, and system design." },
  { year: "2023", event: "Dived deep into LLM agent frameworks and vector databases." },
  { year: "2022", event: "Built several side projects with React and Node.js." },
  { year: "2021", event: "Started exploring eBPF and cloud-native observability." },
  { year: "2020", event: "First line of production code. Been coding ever since." },
];

const writingTopics = [
  "Frontend Architecture", "AI Agent Systems", "Database Internals",
  "WebAssembly", "eBPF & Observability", "Developer Experience",
  "Performance Optimization", "System Design",
];

const values = [
  { icon: "⚡", title: "Speed", desc: "Fast load times, fast feedback loops, fast iteration. Performance is a feature." },
  { icon: "🔧", title: "Craft", desc: "Clean code, clear explanations, thoughtful design. Care about the details." },
  { icon: "🌱", title: "Growth", desc: "Always learning, always building. Share what you learn along the way." },
  { icon: "🔭", title: "Clarity", desc: "Make complex things simple. Good abstraction is everything." },
];

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)]">
      {/* Nav */}
      <nav className="nav fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2">
            <span className="text-base">✦</span>
            <span className="font-semibold text-sm tracking-tight">biluo</span>
          </a>
        </div>
      </nav>

      <section className="pt-32 pb-16 px-6">
        <div className="max-w-3xl mx-auto">

          {/* Hero */}
          <div className="flex items-start gap-8 mb-14 animate-fade-up">
            <div className="flex-shrink-0">
              <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-[#c96438] via-[#d4a574] to-[#e07a4f] flex items-center justify-center text-4xl text-white font-black select-none">
                ✦
              </div>
            </div>
            <div>
              <h1 className="text-4xl font-black tracking-tight mb-2">biluo</h1>
              <p className="text-[var(--color-text-2)] text-base mb-3">
                Frontend Engineer · System Designer · AI Explorer
              </p>
              <p className="text-sm text-[var(--color-text-2)] leading-relaxed max-w-lg">
                Building things with code since the jQuery era. Passionate about developer experience,
                performance, and making complex systems feel simple. Currently diving deep into AI agents,
                WebAssembly, and eBPF. Writing about it all here.
              </p>
            </div>
          </div>

          {/* Values */}
          <div className="mb-10 animate-fade-up" style={{ animationDelay: "40ms" }}>
            <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-5">Principles</h2>
            <div className="grid grid-cols-2 gap-4">
              {values.map(v => (
                <div key={v.title} className="card rounded-2xl p-5">
                  <span className="text-xl mb-3 block">{v.icon}</span>
                  <h3 className="font-bold text-sm mb-1">{v.title}</h3>
                  <p className="text-xs text-[var(--color-text-2)] leading-relaxed">{v.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Writing topics */}
          <div className="mb-10 animate-fade-up" style={{ animationDelay: "80ms" }}>
            <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-5">Writing Topics</h2>
            <div className="card rounded-2xl p-6">
              <div className="flex flex-wrap gap-2">
                {writingTopics.map(topic => (
                  <span key={topic} className="tag">{topic}</span>
                ))}
              </div>
            </div>
          </div>

          {/* Skills */}
          <div className="mb-10 animate-fade-up" style={{ animationDelay: "120ms" }}>
            <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-5">Tech Stack</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {[
                ["Frontend", ["React", "Next.js", "TypeScript", "TailwindCSS", "HTMX", "WebGL"]],
                ["Backend", ["Node.js", "Python", "Go", "PostgreSQL", "Redis", "Nginx"]],
                ["AI & Infra", ["LLM APIs", "VectorDB", "Docker", "Kubernetes", "eBPF", "Prometheus"]],
              ].map(([category, items]) => (
                <div key={category as string} className="card rounded-2xl p-5">
                  <p className="text-xs text-[var(--color-text-2)] font-medium mb-3">{category as string}</p>
                  <div className="flex flex-wrap gap-2">
                    {items.map(item => (
                      <span key={item} className="tag">{item}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Timeline */}
          <div className="mb-10 animate-fade-up" style={{ animationDelay: "160ms" }}>
            <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-5">Timeline</h2>
            <div className="space-y-0">
              {timeline.map((item, i) => (
                <div key={item.year} className="flex gap-5 py-4 border-b border-[var(--color-border)] last:border-0">
                  <span className="text-xs font-mono text-[var(--color-accent)] w-10 flex-shrink-0 pt-0.5">{item.year}</span>
                  <p className="text-sm leading-relaxed">{item.event}</p>
                </div>
              ))}
            </div>
          </div>

          {/* What I Do */}
          <div className="mb-10 animate-fade-up" style={{ animationDelay: "200ms" }}>
            <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-5">What I Do</h2>
            <div className="space-y-4">
              {[
                {
                  title: "Write",
                  desc: "Technical articles on frontend architecture, databases, AI systems, and developer tooling. The blog auto-updates hourly via a cron job that runs the Hexo writer.",
                },
                {
                  title: "Build",
                  desc: "Side projects that scratch an itch — CLI tools, web apps, experimental demos. Most live in the Lab section.",
                },
                {
                  title: "Explore",
                  desc: "Constantly learning: WebAssembly runtime internals, distributed systems patterns, LLM agent frameworks, eBPF for observability.",
                },
              ].map(item => (
                <div key={item.title} className="card rounded-2xl p-6 flex gap-4">
                  <div className="accent-bar w-10 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-bold mb-1">{item.title}</h3>
                    <p className="text-sm text-[var(--color-text-2)] leading-relaxed">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Contact */}
          <div className="animate-fade-up" style={{ animationDelay: "240ms" }}>
            <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium mb-5">Connect</h2>
            <div className="flex flex-wrap gap-3">
              {[
                { label: "GitHub", href: "https://github.com/BiLuoNoBug" },
                { label: "Blog", href: "https://biluonobug.github.io" },
                { label: "Tools", href: "https://biluonobug.github.io/tools" },
                { label: "Lab", href: "https://biluonobug.github.io/lab" },
              ].map(item => (
                <a
                  key={item.label}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium bg-[var(--color-surface)] border border-[var(--color-border)] hover:border-[var(--color-accent)] transition-all"
                >
                  {item.label} →
                </a>
              ))}
            </div>
          </div>

        </div>
      </section>

      {/* Footer */}
      <footer className="footer mt-20 py-12 text-center px-6">
        <p className="text-xs text-[var(--color-text-2)] opacity-50">✦ biluo · biluonobug.github.io</p>
      </footer>
    </main>
  );
}