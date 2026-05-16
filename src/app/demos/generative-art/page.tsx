import styles from './page.module.css'

export const metadata = {
  title: '生成式艺术画板 - Generative Art',
  description: '用算法生成谢尔宾斯基三角形/分形/混沌吸引子 - 多种分形艺术实时渲染',
}

export default function GenerativeArtPage() {
  return (
    <main className={styles.main}>
      <div className={styles.demoHeader}>
        <span className={styles.badge}>🎨 生成艺术</span>
        <h1 className={styles.title}>生成式艺术画板</h1>
        <p className={styles.desc}>
          用算法生成谢尔宾斯基三角形、巴恩斯基蕨、洛伦兹吸引子等经典分形结构。
          移动鼠标与 Julia 集合实时互动，体验数学之美。
        </p>
        <div className={styles.tags}>
          <span>分形</span>
          <span>算法</span>
          <span>Canvas</span>
          <span>数学艺术</span>
        </div>
      </div>
      
      <div className={styles.demoWrapper}>
        <iframe
          src="/demos-content/generative-art/index.html"
          className={styles.iframe}
          title="生成式艺术画板"
          loading="lazy"
        />
      </div>
      
      <div className={styles.backLink}>
        <a href="/demos">← 返回所有 Demo</a>
      </div>
    </main>
  )
}