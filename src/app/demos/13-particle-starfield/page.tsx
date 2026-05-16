import styles from './page.module.css'

export const metadata = {
  title: 'Three.js 3D 粒子星空 - 穿越浩瀚宇宙',
  description: '浩瀚宇宙粒子系统，Three.js 驱动。支持星云/星系/混沌三种模式，鼠标拖拽控制穿越方向，可调节粒子数量、星体大小、飞行速度。',
}

export default function ParticleStarfieldPage() {
  return (
    <main className={styles.main}>
      <div className={styles.demoHeader}>
        <span className={styles.badge}>🌌 Three.js · 3D · WebGL</span>
        <h1 className={styles.title}>Three.js 3D 粒子星空</h1>
        <p className={styles.desc}>
          浩瀚宇宙粒子系统。鼠标拖拽控制飞行方向，支持星云/星系/混沌三种模式，
          可调节粒子数量（最高 2 万）、星体大小、飞行速度，实时 FPS 显示。
        </p>
        <div className={styles.tags}>
          <span>Three.js</span>
          <span>3D</span>
          <span>粒子</span>
          <span>WebGL</span>
          <span>粒子系统</span>
        </div>
      </div>
      
      <div className={styles.demoWrapper}>
        <iframe
          src="/demos-content/13-particle-starfield/index.html"
          className={styles.iframe}
          title="Three.js 3D 粒子星空"
          loading="lazy"
        />
      </div>
      
      <div className={styles.backLink}>
        <a href="/demos">← 返回所有 Demo</a>
      </div>
    </main>
  )
}