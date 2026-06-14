import type { CSSProperties, ReactNode } from 'react';

interface ContainerProps {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}

export default function Container({ children, className = '', style }: ContainerProps) {
  return (
    <div
      className={`container ${className}`}
      style={{
        width: '100%',
        maxWidth: 'var(--container-w)',
        margin: '0 auto',
        padding: '0 24px',
        ...style,
      }}
    >
      {children}
    </div>
  );
}
