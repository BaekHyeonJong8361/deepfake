import styles from './AuthBackground.module.css';

export default function AuthBackground() {
  return (
    <>
      <div className={styles.dotGrid} />
      <div className={`${styles.glow} ${styles.glowTopLeft}`} />
      <div className={`${styles.glow} ${styles.glowBottomRight}`} />
      <div className={`${styles.glow} ${styles.glowCenter}`} />
    </>
  );
}
