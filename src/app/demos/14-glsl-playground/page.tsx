import styles from './page.module.css'

export const metadata = {
  title: 'GLSL Shader Playground - 实时 GLSL 着色器编辑器',
  description: '实时 GLSL 片段着色器编辑器，左侧写代码右侧实时预览，支持等离子、时空扭曲、分形等预设，鼠标交互控制效果',
}

export default function GlslPlaygroundPage() {
  return (
    <main className={styles.main}>
      <div className={styles.demoHeader}>
        <span className={styles.badge}>✨ WebGL 可视化</span>
        <h1 className={styles.title}>GLSL Shader Playground</h1>
        <p className={styles.desc}>
          实时 GLSL 片段着色器编辑器。左侧编写代码，右侧即时预览效果。支持等离子、时空扭曲、分形、水波、粒子漩涡等预设，
          鼠标移动控制光效，实时调整颜色相变、复杂度、缩放、扭曲强度。
        </p>
        <div className={styles.tags}>
          <span>GLSL</span>
          <span>Shader</span>
          <span>WebGL</span>
          <span>实时渲染</span>
          <span>前端</span>
        </div>
      </div>
      
      <div className={styles.demoWrapper}>
        <iframe
          src="/demos-content/14-glsl-playground/index.html"
          className={styles.iframe}
          title="GLSL Shader Playground"
          loading="lazy"
        />
      </div>
      
      <div className={styles.backLink}>
        <a href="/demos">← 返回所有 Demo</a>
      </div>
    </main>
  )
}