import Container from '../common/Container';
import styles from './ServicesSection.module.css';

const services = [
  {
    title: 'Deepfake 탐지',
    desc: '얼굴 합성, 립싱크, 재연. CNN+LSTM 기반 시간축 비일관성 분석.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 2a10 10 0 1 0 10 10H12V2Z" />
        <path d="M12 12 2.1 14.9" />
      </svg>
    ),
  },
  {
    title: 'AI 생성 영상 탐지',
    desc: 'Sora/Runway/Kling 등 T2V 모델. VideoMAE 기반 픽셀 패턴 분석.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
        <line x1="7" y1="2" x2="7" y2="22" />
        <line x1="17" y1="2" x2="17" y2="22" />
        <line x1="2" y1="12" x2="22" y2="12" />
        <line x1="2" y1="7" x2="7" y2="7" />
        <line x1="2" y1="17" x2="7" y2="17" />
        <line x1="17" y1="17" x2="22" y2="17" />
        <line x1="17" y1="7" x2="22" y2="7" />
      </svg>
    ),
  },
  {
    title: 'Grad-CAM 리포트',
    desc: '"가짜입니다"로 끝나지 않습니다. 어떤 프레임의 영역이 의심되는지 시각화.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
  },
  {
    title: '퀴즈 & 통계',
    desc: '미디어 리터러시 자가 진단 테스트 및 글로벌 AI 탐지 동향 통계 제공.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </svg>
    ),
  },
];

export default function ServicesSection() {
  return (
    <section id="s3-services" className={styles.section}>
      <Container>
        <div className={styles.header}>
          <span className={styles.secNumber}>2.0</span>
          <h2 className={styles.secTitle}>우리가 해결합니다</h2>
        </div>
        <div className={styles.grid}>
          {services.map((s) => (
            <div key={s.title} className={styles.card}>
              <div className={styles.icon}>{s.icon}</div>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </div>
          ))}
        </div>
      </Container>
    </section>
  );
}
