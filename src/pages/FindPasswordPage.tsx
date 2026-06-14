import { useState } from 'react';
import { Link } from 'react-router-dom';
import AuthBackground from '../components/common/AuthBackground';
import s from '../styles/auth.module.css';
import ls from './LoginPage.module.css';

export default function FindPasswordPage() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Reset password:', { username, email });
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

          <h1>비밀번호 찾기</h1>
          <p className={ls.subtitle}>가입 시 등록한 아이디와 이메일을 입력해주세요.</p>
        </div>

        <form className={ls.form} onSubmit={handleSubmit}>
          <div className={s.formGroupLg}>
            <label htmlFor="fp-username" className={s.formLabel}>아이디</label>
            <div className={s.inputWrapper}>
              <input
                type="text" id="fp-username"
                placeholder="아이디를 입력하세요"
                value={username} onChange={(e) => setUsername(e.target.value)}
              />
            </div>
          </div>

          <div className={s.formGroupLg}>
            <label htmlFor="fp-email" className={s.formLabel}>이메일 주소</label>
            <div className={s.inputWrapper} style={{ display: 'flex' }}>
              <input
                type="email" id="fp-email"
                placeholder="example@domain.com"
                value={email} onChange={(e) => setEmail(e.target.value)}
              />
              <button type="button" className={s.btnSendCode}>인증번호 발송</button>
            </div>
          </div>

          <div className={s.formGroupLg}>
            <label htmlFor="fp-code" className={s.formLabel}>인증번호</label>
            <div className={s.inputWrapper}>
              <input
                type="text" id="fp-code"
                placeholder="이메일로 전송된 인증번호 입력"
                value={code} onChange={(e) => setCode(e.target.value)}
              />
            </div>
          </div>

          <div className={s.formGroupLg}>
            <label htmlFor="fp-new-pw" className={s.formLabel}>새 비밀번호</label>
            <div className={s.inputWrapper}>
              <input
                type="password" id="fp-new-pw"
                placeholder="새로운 비밀번호 입력"
                value={newPw} onChange={(e) => setNewPw(e.target.value)}
              />
            </div>
          </div>

          <div className={s.formGroupLg}>
            <label htmlFor="fp-confirm-pw" className={s.formLabel}>새 비밀번호 확인</label>
            <div className={s.inputWrapper}>
              <input
                type="password" id="fp-confirm-pw"
                placeholder="새로운 비밀번호 재입력"
                value={confirmPw} onChange={(e) => setConfirmPw(e.target.value)}
              />
            </div>
          </div>

          <button type="submit" className={ls.submitBtn}>비밀번호 재설정</button>
        </form>

        <div className={s.authFooter} style={{ marginTop: 24 }}>
          기억나셨나요? <Link to="/login">로그인하러 가기</Link>
        </div>
      </div>
    </div>
  );
}
