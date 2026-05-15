import { posts } from "@/lib/posts/posts";
import Link from "next/link";
import ThemeToggle from "@/components/ThemeToggle";

function FeaturedCard({ post }: { post: typeof posts[0] }) {
  return (
    <Link href={`/posts/${post.slug}`} className="group block">
      <article className="card rounded-3xl overflow-hidden hover:-translate-y-1">
        <div className="h-1 bg-gradient-to-r from-[#c96438] via-[#d4a574] to-[#e07a4f]" />
        <div className="p-10 md:p-14">
          <div className="flex items-center gap-2 flex-wrap mb-5">
            <span className="text-xs text-[var(--color-text-2)] font-mono">{post.date}</span>
            {post.tags.slice(0, 3).map(t => <span key={t} className="tag">{t}</span>)}
          </div>
          <h2 className="text-3xl md:text-5xl font-black leading-tight tracking-tight mb-5 text-[var(--color-text)] group-hover:text-[var(--color-accent)] transition-colors">
            {post.title}
          </h2>
          <p className="text-[var(--color-text-2)] text-base leading-relaxed max-w-2xl mb-8">
            {post.excerpt}
          </p>
          <div className="flex items-center gap-5">
            <span className="text-sm font-medium text-[var(--color-accent)] group-hover:underline underline-offset-4">
              Read full article →
            </span>
            <span className="text-xs text-[var(--color-text-2)]">{post.content.length} words</span>
          </div>
        </div>
      </article>
    </Link>
  );
}

function PostCard({ post, i }: { post: typeof posts[0]; i: number }) {
  return (
    <Link href={`/posts/${post.slug}`} className="group block" style={{ animationDelay: `${i * 60}ms` }}>
      <article className="card rounded-2xl p-7 h-full flex flex-col">
        <div className="accent-bar w-10 mb-6" />
        <div className="flex items-center gap-2 flex-wrap mb-4">
          <span className="text-xs text-[var(--color-text-2)] font-mono">{post.date}</span>
          {post.tags.slice(0, 2).map(t => <span key={t} className="tag">{t}</span>)}
        </div>
        <h3 className="text-base font-bold mb-3 leading-snug text-[var(--color-text)] group-hover:text-[var(--color-accent)] transition-colors line-clamp-2">
          {post.title}
        </h3>
        <p className="text-[var(--color-text-2)] text-sm leading-relaxed line-clamp-3 mb-5 flex-1">
          {post.excerpt}
        </p>
        <div className="flex items-center justify-between mt-auto">
          <span className="text-xs text-[var(--color-accent)] opacity-60 group-hover:opacity-100 transition-opacity">Read →</span>
          <span className="text-xs text-[var(--color-text-2)]">{post.content.length}w</span>
        </div>
      </article>
    </Link>
  );
}

export default function Home() {
  const featured = posts[0];
  const rest = posts.slice(1);

  return (
    <main className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)]">
      {/* Nav */}
      <nav className="nav fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-base">✦</span>
            <span className="font-semibold text-sm tracking-tight">biluo</span>
          </div>
          <div className="flex items-center gap-3">
            <a href="/about" className="text-xs text-[var(--color-text-2)] hover:text-[var(--color-text)] transition-colors">About</a>
            <a href="/tools" className="text-xs text-[var(--color-text-2)] hover:text-[var(--color-text)] transition-colors">Tools</a>
            <a href="/lab" className="text-xs text-[var(--color-text-2)] hover:text-[var(--color-text)] transition-colors">Lab</a>
            <a href="/dashboard" className="text-xs text-[var(--color-text-2)] hover:text-[var(--color-text)] transition-colors">Dashboard</a>
            <span className="text-xs text-[var(--color-text-2)] opacity-40">|</span>
            <span className="text-xs text-[var(--color-text-2)]">{posts.length} articles</span>
            <ThemeToggle />
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative min-h-[75vh] flex items-center justify-center text-center px-6 overflow-hidden">
        <div className="absolute blob-1 pointer-events-none"
          style={{ top: "-15%", left: "-8%", width: 520, height: 520, borderRadius: "50%", background: "radial-gradient(circle, rgba(201,100,56,0.13) 0%, transparent 70%)" }}
        />
        <div className="absolute blob-2 pointer-events-none"
          style={{ bottom: "-15%", right: "-6%", width: 480, height: 480, borderRadius: "50%", background: "radial-gradient(circle, rgba(212,165,116,0.12) 0%, transparent 70%)" }}
        />
        <div className="absolute blob-3 pointer-events-none"
          style={{ top: "35%", right: "18%", width: 280, height: 280, borderRadius: "50%", background: "radial-gradient(circle, rgba(201,100,56,0.08) 0%, transparent 70%)" }}
        />

        <div className="relative z-10 max-w-3xl mx-auto">
          <div className="mb-10 text-sm text-[var(--color-text-2)] animate-fade-up">
            <span className="inline-flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse" />
              personal technical blog
            </span>
          </div>

          <h1 className="text-5xl md:text-7xl font-black mb-8 leading-[1.02] tracking-tight text-[var(--color-text)] animate-fade-up" style={{ animationDelay: "80ms" }}>
            writing on
            <br />
            <span className="text-[var(--color-accent)]">engineering</span>
          </h1>

          <p className="text-base text-[var(--color-text-2)] leading-relaxed animate-fade-up" style={{ animationDelay: "160ms" }}>
            frontend · backend · databases · AI · system design
            <br />
            <span className="text-xs opacity-50">auto-updated</span>
          </p>

          <div className="mt-10 animate-fade-up" style={{ animationDelay: "240ms" }}>
            <a href="#posts"
              className="inline-flex items-center gap-2.5 px-8 py-4 rounded-2xl font-semibold text-sm bg-[var(--color-text)] text-[var(--color-bg)] hover:opacity-90 transition-all duration-300 hover:-translate-y-0.5">
              read all articles →
            </a>
          </div>
        </div>
      </section>

      {/* Featured */}
      {featured && (
        <section className="max-w-7xl mx-auto px-6 py-10">
          <div className="flex items-center gap-4 mb-6">
            <span className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium">Featured</span>
            <div className="h-px flex-1 bg-[var(--color-border)]" />
          </div>
          <FeaturedCard post={featured} />
        </section>
      )}

      {/* All posts */}
      <section id="posts" className="max-w-7xl mx-auto px-6 py-10">
        <div className="flex items-center gap-4 mb-8">
          <span className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium">All articles</span>
          <div className="h-px flex-1 bg-[var(--color-border)]" />
          <span className="text-xs text-[var(--color-text-2)]">{posts.length}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {rest.map((post, i) => (
            <PostCard key={post.slug} post={post} i={i} />
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="footer mt-20 py-12 text-center">
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs text-[var(--color-text-2)]">✦ biluo</span>
          <p className="text-xs text-[var(--color-text-2)] opacity-50">biluonobug.github.io</p>
        </div>
      </footer>
    </main>
  );
}