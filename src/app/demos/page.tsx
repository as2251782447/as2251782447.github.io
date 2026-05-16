import Link from "next/link";

const demos = [
  { id: "3-sql-executor", title: "在线 SQL 执行器", desc: "浏览器里跑 SQLite，支持上传 CSV", tags: ["SQLite", "WASM"], icon: "🗄️", color: "#60a5fa" },
  { id: "13-particle-starfield", title: "Three.js 3D 粒子星空", desc: "浩瀚宇宙粒子系统，鼠标控制相机穿越星海", tags: ["Three.js", "3D", "粒子"], icon: "✨", color: "#facc15" },
  { id: "14-glsl-playground", title: "GLSL Shader Playground", desc: "实时 GLSL 片段着色器编辑器，左侧写代码右侧看效果", tags: ["GLSL", "Shader", "WebGL"], icon: "🎨", color: "#f472b6" },
  { id: "15-tensorflow-classification", title: "TensorFlow.js 图像分类", desc: "浏览器里跑 MobileNet 实时分类摄像头画面", tags: ["TensorFlow.js", "ML"], icon: "🤖", color: "#fb923c" },
  { id: "18-plotly-realtime", title: "Plotly 实时图表", desc: "股票/传感器数据实时折线图，支持缩放和拖拽", tags: ["Plotly", "图表"], icon: "📈", color: "#4ade80" },
  { id: "generative-art", title: "生成式艺术画板", desc: "用算法生成谢尔宾斯基三角形/分形/混沌吸引子", tags: ["生成艺术", "分形"], icon: "🌀", color: "#a78bfa" },
];

export const metadata = {
  title: "Demos · biluo",
  description: "Interactive demos, tools, and creative experiments.",
};

export default function DemosPage() {
  return (
    <main className="max-w-5xl mx-auto px-6 py-32">
      <div className="mb-12">
        <h1 className="text-4xl font-black mb-4">Demos</h1>
        <p className="text-[var(--color-text-2)] text-base">Interactive tools, creative experiments, and live demos.</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {demos.map((d) => (
          <Link key={d.id} href={`/demos/${d.id}`} className="card rounded-2xl p-7 block group">
            <div className="flex items-center gap-3 mb-5">
              <span className="text-3xl">{d.icon}</span>
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: d.color }}
              />
            </div>
            <h2 className="text-base font-bold mb-3 group-hover:text-[var(--color-accent)] transition-colors">{d.title}</h2>
            <p className="text-[var(--color-text-2)] text-sm leading-relaxed mb-5">{d.desc}</p>
            <div className="flex flex-wrap gap-2">
              {d.tags.map((t) => (
                <span key={t} className="text-xs px-2 py-1 rounded-full bg-[var(--color-bg-2)]">{t}</span>
              ))}
            </div>
          </Link>
        ))}
      </div>
    </main>
  );
}