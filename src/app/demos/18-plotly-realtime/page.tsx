import styles from './page.module.css'

export const metadata = {
  title: 'Plotly 实时图表 - 传感器数据可视化',
  description: '多源传感器数据实时可视化，支持缩放、拖拽、筛选。Plotly.js 驱动，支持温度/湿度/气压多通道实时图表。',
}

export default function PlotlyRealtimePage() {
  return (
    <main className={styles.main}>
      <div className={styles.demoHeader}>
        <span className={styles.badge}>📊 数据可视化</span>
        <h1 className={styles.title}>Plotly 实时图表</h1>
        <p className={styles.desc}>
          多源传感器数据实时可视化。支持温度、湿度、气压三通道数据实时绘制，
          可调节更新频率、噪声等级、曲线样式，完美模拟工业传感器监控场景。
        </p>
        <div className={styles.tags}>
          <span>Plotly</span>
          <span>实时图表</span>
          <span>传感器数据</span>
          <span>数据可视化</span>
        </div>
      </div>
      
      <div className={styles.demoWrapper}>
        <iframe
          src="/demos-content/18-plotly-realtime/index.html"
          className={styles.iframe}
          title="Plotly 实时图表"
          loading="lazy"
        />
      </div>
      
      <div className={styles.backLink}>
        <a href="/demos">← 返回所有 Demo</a>
      </div>
    </main>
  )
}