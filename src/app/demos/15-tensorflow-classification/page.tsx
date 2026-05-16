import styles from './page.module.css'

export const metadata = {
  title: 'TensorFlow.js 图像分类 - 实时摄像头识别',
  description: '浏览器里跑 MobileNet 实时分类摄像头画面 - TensorFlow.js + WebRTC 实现',
}

export default function TensorFlowClassificationPage() {
  return (
    <main className={styles.main}>
      <div className={styles.demoHeader}>
        <span className={styles.badge}>🧠 机器学习</span>
        <h1 className={styles.title}>TensorFlow.js 实时图像分类</h1>
        <p className={styles.desc}>
          在浏览器里跑 MobileNet v2 实时分类摄像头画面。无需服务器，直接在网页上运行深度学习模型。
          支持切换前后摄像头，实时显示 FPS 和置信度。
        </p>
        <div className={styles.tags}>
          <span>TensorFlow.js</span>
          <span>MobileNet</span>
          <span>计算机视觉</span>
          <span>WebRTC</span>
        </div>
      </div>
      
      <div className={styles.demoWrapper}>
        <iframe
          src="/demos-content/15-tensorflow-classification/index.html"
          className={styles.iframe}
          title="TensorFlow.js 图像分类"
          loading="lazy"
        />
      </div>
      
      <div className={styles.backLink}>
        <a href="/demos">← 返回所有 Demo</a>
      </div>
    </main>
  )
}