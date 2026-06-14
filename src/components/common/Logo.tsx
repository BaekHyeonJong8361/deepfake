import { Link } from 'react-router-dom';
import styles from './Logo.module.css';

export default function Logo() {
  return (
    <Link to="/" className={styles.logo}>
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="24" height="24">
        <circle cx="9" cy="12" r="7" stroke="currentColor" strokeWidth="2" />
        <circle cx="15" cy="12" r="7" stroke="currentColor" strokeWidth="2" strokeDasharray="4 4" />
      </svg>
      Solomon AI
    </Link>
  );
}
