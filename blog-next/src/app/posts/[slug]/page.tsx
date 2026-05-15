import { posts } from "@/lib/posts/posts";
import Link from "next/link";

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({ params }: Props) {
  const { slug } = await params;
  const post = posts.find(p => p.slug === slug);
  if (!post) return { title: "Not found" };
  return { title: `${post.title} · biluo`, description: post.excerpt };
}

export function generateStaticParams() {
  return posts.map(p => ({ slug: p.slug }));
}

const tags: Record<string, string> = {
  AI: "AI",
  前端: "前端",
  后端: "后端",
  数据库: "数据库",
  架构: "架构",
};

function renderContent(content: string) {
  const lines = content.split("\n");
  const elems: string[] = [];
  let i = 0;
  let inList = false;

  while (i < lines.length) {
    const line = lines[i].trim();

    if (line.startsWith("## ")) {
      if (inList) { elems.push("</ul>"); inList = false; }
      elems.push(`<h2 class="text-2xl font-black text-[var(--color-text)] mt-12 mb-4 tracking-tight border-b border-[var(--color-border)] pb-3">${line.slice(3)}</h2>`);
    } else if (line.startsWith("### ")) {
      if (inList) { elems.push("</ul>"); inList = false; }
      elems.push(`<h3 class="text-base font-bold text-[var(--color-text)] mt-8 mb-3">${line.slice(4)}</h3>`);
    } else if (line.startsWith("|")) {
      if (inList) { elems.push("</ul>"); inList = false; }
      const cells = line.split("|").filter(c => c.trim() && !c.trim().match(/^[-: ]+$/));
      if (cells.length > 0) {
        elems.push(`<div class="flex gap-3 my-2 text-sm text-[var(--color-text-2)]"><span class="flex-1">${cells.join("</span><span class='flex-1'>")}</span></div>`);
      }
    } else if (line.match(/^[-*] /)) {
      if (!inList) { elems.push("<ul class='space-y-2 my-5 ml-4 list-disc'>"); inList = true; }
      elems.push(`<li class="text-[var(--color-text-2)] leading-relaxed">${line.slice(2)}</li>`);
    } else if (!line) {
      if (inList) { elems.push("</ul>"); inList = false; }
    } else {
      if (inList) { elems.push("</ul>"); inList = false; }
      const text = line
        .replace(/\*\*(.+?)\*\*/g, "<strong class='text-[var(--color-text)] font-semibold'>$1</strong>")
        .replace(/`(.+?)`/g, "<code class='bg-[var(--color-surface)] text-[var(--color-accent)] px-1.5 py-0.5 rounded text-sm font-mono border border-[var(--color-border)]'>$1</code>");
      elems.push(`<p class="text-[var(--color-text-2)] leading-relaxed my-4">${text}</p>`);
    }
    i++;
  }
  if (inList) elems.push("</ul>");
  return elems.join("");
}

export default async function PostPage({ params }: Props) {
  const { slug } = await params;
  const post = posts.find(p => p.slug === slug);
  if (!post) return (
    <main className="min-h-screen bg-[var(--color-bg)] flex items-center justify-center">
      <p className="text-[var(--color-text-2)]">Not found</p>
    </main>
  );

  const others = posts.filter(p => p.slug !== slug).slice(0, 3);

  return (
    <main className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)] transition-colors">
      {/* Nav */}
      <nav className="nav fixed top-0 left-0 right-0 z-50">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 group">
            <span className="text-base">✦</span>
            <span className="font-semibold text-sm tracking-tight group-hover:text-[var(--color-accent)] transition-colors">biluo</span>
          </Link>
          <Link href="/" className="text-sm text-[var(--color-text-2)] hover:text-[var(--color-text)] transition-colors">
            ← All articles
          </Link>
        </div>
      </nav>

      {/* Article */}
      <article className="max-w-4xl mx-auto px-6 pt-36 pb-20">
        {/* Header */}
        <header className="mb-12">
          <div className="flex items-center gap-2 flex-wrap mb-5">
            <span className="text-xs text-[var(--color-text-2)] font-mono">{post.date}</span>
            {post.tags.map(t => (
              <span key={t} className="tag">{t}</span>
            ))}
          </div>

          <h1 className="text-4xl md:text-5xl font-black leading-[1.05] tracking-tight mb-5">
            {post.title}
          </h1>

          <p className="text-base text-[var(--color-text-2)] leading-relaxed mb-10">
            {post.excerpt}
          </p>

          <div className="flex items-center gap-3 text-xs text-[var(--color-text-2)] pb-8 border-b border-[var(--color-border)]">
            <span>✦</span>
            <span>biluo</span>
            <span>·</span>
            <span>{post.content.length} words</span>
          </div>
        </header>

        {/* Body */}
        <div className="text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: renderContent(post.content) }} />

        {/* Back */}
        <div className="mt-16 pt-8 border-t border-[var(--color-border)]">
          <Link href="/" className="inline-flex items-center gap-2 text-sm font-medium text-[var(--color-accent)] hover:underline underline-offset-4 transition-colors">
            ← All articles
          </Link>
        </div>

        {/* More */}
        {others.length > 0 && (
          <div className="mt-20">
            <div className="flex items-center gap-3 mb-8">
              <span className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium">More reading</span>
              <div className="h-px flex-1 bg-[var(--color-border)]" />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {others.map(p => (
                <Link key={p.slug} href={`/posts/${p.slug}`} className="group">
                  <article className="card rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-xs text-[var(--color-text-2)] font-mono">{p.date}</span>
                      {p.tags[0] && <span className="tag">{p.tags[0]}</span>}
                    </div>
                    <h4 className="text-sm font-bold text-[var(--color-text)] group-hover:text-[var(--color-accent)] transition-colors leading-snug line-clamp-2">
                      {p.title}
                    </h4>
                  </article>
                </Link>
              ))}
            </div>
          </div>
        )}
      </article>

      {/* Footer */}
      <footer className="footer py-10 text-center">
        <div className="flex flex-col items-center gap-1.5">
          <span className="text-xs text-[var(--color-text-2)]">✦ biluo</span>
          <p className="text-xs text-[var(--color-text-2)] opacity-50">biluonobug.github.io</p>
        </div>
      </footer>
    </main>
  );
}