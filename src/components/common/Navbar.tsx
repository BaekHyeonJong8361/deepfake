import { Link } from 'react-router-dom';
import Logo from './Logo';
import Container from './Container';
import styles from './Navbar.module.css';

interface NavbarProps {
  variant?: 'scan' | 'home';
}

export default function Navbar({ variant = 'scan' }: NavbarProps) {
  return (
    <Container>
      <header className={styles.header}>
        <Logo />
        <nav className={styles.navLinks}>
          {variant === 'home' ? (
            <>
              <Link to="/">Scan</Link>
              <Link to="/login">Login</Link>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
              <a href="#s2-problem">About</a>
              <a href="#s4-cta">Contact</a>
            </>
          ) : (
            <>
              <Link to="/home">Home</Link>
              <Link to="/login">Login</Link>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
            </>
          )}
        </nav>
      </header>
    </Container>
  );
}
