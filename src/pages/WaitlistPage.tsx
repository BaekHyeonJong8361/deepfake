import { Link } from 'react-router-dom';
import AuthBackground from '../components/common/AuthBackground';
import s from '../styles/auth.module.css';
import ls from './LoginPage.module.css';

export default function WaitlistPage() {
  return (
    <div className={ls.page}>
      <AuthBackground />

      <div className={ls.card} style={{ maxWidth: 560, padding: '64px 48px', textAlign: 'center' as const }}>
        <div className={s.iconWrapper}>
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
        </div>
        <h1 style={{ fontSize: '2rem', fontWeight: 600, letterSpacing: '-0.03em', marginBottom: 16 }}>
          가입 신청이 완료되었습니다
        </h1>
        <p className={s.desc}>
          Solomon AI 베타 서비스에 관심 가져주셔서 감사합니다.<br />
          현재 순차적으로 승인 절차를 진행하고 있습니다.
        </p>

        <div className={s.noticeBox}>
          <div className={s.noticeTitle}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            승인 안내
          </div>
          <p className={s.noticeText}>
            가입 시 입력하신 이메일로 승인 결과가 안내될 예정입니다.<br />
            대기 인원에 따라 영업일 기준 1~3일 정도 소요될 수 있습니다.
          </p>
        </div>

        <Link to="/" className={s.btnOutline}>메인으로 돌아가기</Link>
      </div>
    </div>
  );
}
