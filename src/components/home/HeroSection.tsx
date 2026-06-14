import { useNavigate } from 'react-router-dom';
import Button from '../common/Button';
import Container from '../common/Container';
import WireframeWave from './WireframeWave';
import styles from './HeroSection.module.css';

export default function HeroSection() {
  const navigate = useNavigate();

  return (
    <section className={styles.hero}>
      <Container>
        <div className={styles.grid}>
          <div className={styles.text}>
            <h2 className={styles.title}>
              Detect What Your Eyes Can't<span className={styles.dot}>.</span>
            </h2>
            <p className={styles.korSub}>
              사람의 눈으로는 더 이상 구별할 수 없습니다.
              <br />
              AI가 만든 영상, AI가 찾아냅니다.
            </p>
            <p className={styles.desc}>
              CNN-LSTM 시간축 분석 + Grad-CAM 시각적 설명까지. 결과만 던지지 않습니다 — '왜 가짜인지' 보여드립니다.
            </p>
            <div className={styles.actions}>
              <Button variant="dark" onClick={() => navigate('/')}>
                지금 바로 분석하기
              </Button>
              <Button variant="outline" onClick={() => {
                document.getElementById('s3-services')?.scrollIntoView({ behavior: 'smooth' });
              }}>
                서비스 소개 보기
              </Button>
            </div>
          </div>

          <div className={styles.graphic}>
            <WireframeWave />
          </div>
        </div>
      </Container>
    </section>
  );
}
