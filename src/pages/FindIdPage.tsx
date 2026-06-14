import { useState } from 'react';
import { Link } from 'react-router-dom';
import AuthBackground from '../components/common/AuthBackground';
import s from '../styles/auth.module.css';
import ls from './LoginPage.module.css';

export default function FindIdPage() {
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Find ID:', { email, code });
  };

  return (
    <div className={ls.page}>
      <AuthBackground />

      <div className={ls.card}>
        {/* Logo */}
        <div className={ls.header}>
          <Link to="/" className={ls.logo}>
            <svg viewBox="0 0 24 24" fill="none" width="32" height="32">
              <circle cx="9" cy="12" r="7" stroke="currentColor" strokeWidth="2.5" />
              <circle cx="15" cy="12" r="7" stroke="currentColor" strokeWidth="2.5" strokeDasharray="4 4" />
            </svg>
            <span>Solomon AI</span>
          </Link>

          <h1>아이디 찾기</h1>
          <p className={ls.subtitle}>가입 시 등록한 이메일 주소를 입력해 주세요.</p>
        </div>

        <form className={ls.form} onSubmit={handleSubmit}>
          <div className={s.formGroupLg}>
            <label htmlFor="find-email" className={s.formLabel}>이메일 주소</label>
            <div className={s.inputRow}>
              <input
                type="email" id="find-email" className={s.formInput}
                placeholder="example@domain.com" required
                value={email} onChange={(e) => setEmail(e.target.value)}
              />
              <button type="button" className={s.btnSecondary}>인증번호 발송</button>
            </div>
          </div>

          <div className={s.formGroupLg}>
            <label htmlFor="find-code" className={s.formLabel}>인증번호</label>
            <input
              type="text" id="find-code" className={s.formInput}
              placeholder="6자리 숫자 입력" required
              value={code} onChange={(e) => setCode(e.target.value)}
            />
          </div>

          <button type="submit" className={ls.submitBtn}>아이디 찾기</button>
        </form>

        <div className={s.authFooterLinks} style={{ marginTop: 24 }}>
          <Link to="/find-password">비밀번호 찾기</Link>
          <span className={s.divider}>|</span>
          <Link to="/login">로그인으로 돌아가기</Link>
        </div>
      </div>
    </div>
  );
}
