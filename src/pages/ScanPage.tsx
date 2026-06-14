import Navbar from '../components/common/Navbar';
import Container from '../components/common/Container';
import BackendHealthCheck from '../components/common/BackendHealthCheck';
import ParticleCanvas from '../components/scan/ParticleCanvas';
import ScanInput from '../components/scan/ScanInput';
import styles from './ScanPage.module.css';

export default function ScanPage() {
  return (
    <section className={styles.scanPage}>
      <ParticleCanvas />
      <Navbar variant="scan" />

      <Container className={styles.content}>
        <h1>Don't Trust Your Eyes.</h1>
        <p className={styles.korSub}>AI가 만든 영상, 3초 만에 판별합니다</p>
        <ScanInput />

        <div className={styles.trustCopy}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
          판별 결과와 '그 이유'까지 설명합니다
        </div>
        <BackendHealthCheck />
      </Container>

      <footer className={styles.footer}>
        <span>Solomon AI by RunningSnail</span>
        <a href="#">About</a> ·{' '}
        <a href="#">Contact</a> ·{' '}
        <a href="#">Blog</a> ·{' '}
        <a href="#">FAQ</a>
      </footer>
    </section>
  );
}
