import styles from './page.module.css'

export const metadata = {
  title: '在线 SQL 执行器 - Browser SQLite',
  description: '浏览器里跑 SQLite，支持上传 CSV、示例数据、历史记录 - 基于 sql.js WASM 实现',
}

export default function SqlExecutorPage() {
  return (
    <main className={styles.main}>
      <div className={styles.demoHeader}>
        <span className={styles.badge}>🗄️ 数据库工具</span>
        <h1 className={styles.title}>在线 SQL 执行器</h1>
        <p className={styles.desc}>
          纯浏览器运行的 SQLite 数据库，无需服务器，支持 CSV 导入、SQL 格式化、历史记录。
          选好示例数据后尝试 JOIN 查询，体验关联数据的威力。
        </p>
        <div className={styles.tags}>
          <span>SQLite</span>
          <span>WASM</span>
          <span>sql.js</span>
          <span>数据分析</span>
        </div>
      </div>
      
      <div className={styles.demoWrapper}>
        <iframe
          src="/demos-content/3-sql-executor/index.html"
          className={styles.iframe}
          title="在线 SQL 执行器"
          loading="lazy"
        />
      </div>
      
      <div className={styles.backLink}>
        <a href="/demos">← 返回所有 Demo</a>
      </div>
    </main>
  )
}