import Container from '../common/Container';
import styles from './ProblemSection.module.css';

const stats = [
  { value: '7.2M', label: '하루 720만 개', desc: '2025년 기준 매일 생성되는 AI 영상', highlight: false },
  { value: '550%', label: '550% 증가', desc: '2019년 대비 딥페이크 증가율 (WEF)', highlight: true },
  { value: '96%', label: '96% 비동의', desc: '딥페이크의 96%는 비동의 포르노그래피', highlight: false },
  { value: '50%', label: '정확도 50%', desc: '일반인 육안 판별 확률 (동전던지기 수준)', highlight: false },
];

export default function ProblemSection() {
  return (
    <section id="s2-problem" className={styles.section}>
      <Container>
        <div className={styles.header}>
          <span className={styles.secNumber}>1.0</span>
          <h2 className={styles.secTitle}>문제는 이미 시작됐습니다</h2>
        </div>
        <div className={styles.grid}>
          {stats.map((s) => (
            <div key={s.value} className={styles.card}>
              <div
                className={styles.statValue}
                style={s.highlight ? { color: 'var(--accent-blue)' } : undefined}
              >
                {s.value}
              </div>
              <p>
                <strong>{s.label}</strong>
                <br />
                {s.desc}
              </p>
            </div>
          ))}
          <div className={styles.closing}>
            Sora, Runway, Kling — 생성 모델은 매달 진화합니다.
            <br />
            판별 도구 없이 맨눈으로 버티는 시대는 끝났습니다.
          </div>
        </div>
      </Container>
    </section>
  );
}
