import { useNavigate } from 'react-router-dom';
import Button from '../common/Button';
import styles from './CtaSection.module.css';

export default function CtaSection() {
  const navigate = useNavigate();

  return (
    <section className={styles.cta}>
      <div className={styles.inner}>
        <span className={styles.secNumber}>3.0</span>
        <h2 className={styles.title}>
          지금 바로 확인해보세요<span className={styles.dot}>.</span>
        </h2>
        <p className={styles.desc}>
          영상 하나면 충분합니다. 링크를 붙여넣거나, 파일을 업로드하세요.
        </p>
        <p className={styles.highlight}>
          로그인 없이 즉시 사용 가능합니다. 회원가입하면 분석 기록 저장 · 히스토리 조회까지.
        </p>
        <Button variant="primary" size="large" onClick={() => navigate('/')}>
          무료로 영상 분석하기
        </Button>
        <div className={styles.footer}>분석은 무료입니다. 가입도 30초면 끝.</div>
      </div>
    </section>
  );
}
