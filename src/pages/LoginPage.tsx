import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import styles from './LoginPage.module.css';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const { login, status, error } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await login({ email, password });
      navigate('/');
    } catch {
      // Error message is managed by the auth store.
    }
  };

  return (
    <div className={styles.page}>
      {/* Background effects */}
      <div className={styles.dotGrid} />
      <div className={`${styles.glow} ${styles.glowTopLeft}`} />
      <div className={`${styles.glow} ${styles.glowBottomRight}`} />
      <div className={`${styles.glow} ${styles.glowCenter}`} />

      {/* Card */}
      <div className={styles.card}>
        {/* Logo */}
        <div className={styles.header}>
          <Link to="/" className={styles.logo}>
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="32" height="32">
              <circle cx="9" cy="12" r="7" stroke="currentColor" strokeWidth="2.5" />
              <circle cx="15" cy="12" r="7" stroke="currentColor" strokeWidth="2.5" strokeDasharray="4 4" />
            </svg>
            <span>Solomon AI</span>
          </Link>

          <h1>Log In</h1>
          <p className={styles.subtitle}>
            Don't have an account?{' '}
            <Link to="/register" className={styles.link}>Sign up for free</Link>
          </p>
          <p className={styles.subtitle} style={{ marginTop: 4 }}>
            <Link to="/find-id" className={styles.link} style={{ fontSize: '0.8rem' }}>아이디 찾기</Link>
          </p>
        </div>

        {/* Form */}
        <form className={styles.form} onSubmit={handleSubmit}>
          {/* Email */}
          <div className={styles.field}>
            <label htmlFor="email">Email</label>
            <div className={styles.inputWrapper}>
              <div className={styles.inputIcon}>
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect width="20" height="16" x="2" y="4" rx="2" />
                  <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
                </svg>
              </div>
              <input
                type="email"
                id="email"
                placeholder="example@mail.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>

          {/* Password */}
          <div className={styles.field}>
            <label htmlFor="password">Password</label>
            <div className={styles.inputWrapper}>
              <div className={styles.inputIcon}>
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                placeholder="********"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                className={styles.togglePassword}
                onClick={() => setShowPassword(!showPassword)}
                aria-label="Toggle password visibility"
              >
                {showPassword ? (
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9.88 9.88a3 3 0 1 0 4.24 4.24" />
                    <path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68" />
                    <path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61" />
                    <line x1="2" y1="2" x2="22" y2="22" />
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                )}
              </button>
            </div>
            <div className={styles.forgotRow}>
              <Link to="/find-password" className={styles.forgotLink}>Forgot your password?</Link>
            </div>
          </div>

          {/* Submit */}
          <button type="submit" className={styles.submitBtn}>
            {status === 'loading' ? 'Logging in...' : 'Log In'}
          </button>
          {error && <p className={styles.subtitle}>{error}</p>}
        </form>
      </div>
    </div>
  );
}
