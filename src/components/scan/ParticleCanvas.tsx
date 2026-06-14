import { useEffect, useRef } from 'react';

const CHAR_SIZE = 12;
const DENSITY_CHARS =
  " .'`^,:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";

function simpleNoise(x: number, y: number, t: number) {
  return (
    Math.sin(x * 0.05 + t) * Math.cos(y * 0.05 + t) +
    Math.sin(x * 0.01 - t) * Math.cos(y * 0.12) * 0.5
  );
}

export default function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let width = 0;
    let height = 0;
    let time = 0;
    const mousePos = { x: 0, y: 0 };

    function resize() {
      const parent = canvas!.parentElement;
      if (!parent) return;
      width = parent.clientWidth;
      height = parent.clientHeight;

      const dpr = window.devicePixelRatio || 1;
      canvas!.width = width * dpr;
      canvas!.height = height * dpr;
      ctx!.scale(dpr, dpr);
      canvas!.style.width = width + 'px';
      canvas!.style.height = height + 'px';
    }

    function render() {
      ctx!.clearRect(0, 0, width, height);
      ctx!.font = `${CHAR_SIZE}px monospace`;
      ctx!.textAlign = 'center';
      ctx!.textBaseline = 'middle';

      const colsCount = Math.ceil(width / CHAR_SIZE);
      const rowsCount = Math.ceil(height / CHAR_SIZE);

      for (let y = 0; y < rowsCount; y++) {
        // Skip top 30% for cleaner header area
        if (y < rowsCount * 0.3) continue;

        for (let x = 0; x < colsCount; x++) {
          const posX = x * CHAR_SIZE;
          const posY = y * CHAR_SIZE;

          const dx = posX - mousePos.x;
          const dy = posY - mousePos.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          const normalizedY = (rowsCount - y) / rowsCount;

          const noiseVal = simpleNoise(x, y, time * 0.5);
          const mountainHeight =
            0.3 +
            Math.sin(x * 0.05 + time * 0.1) * 0.1 +
            Math.cos(x * 0.2) * 0.05;

          let char = '';
          let alpha = 0;

          if (normalizedY < mountainHeight + noiseVal * 0.1) {
            const index = Math.floor(
              Math.abs(noiseVal) * DENSITY_CHARS.length,
            );
            char = DENSITY_CHARS[index % DENSITY_CHARS.length];
            alpha = 1 - normalizedY * 2;
          }

          if (dist < 150) {
            const lensStrength = 1 - dist / 150;

            if (Math.random() > 0.5) {
              char = Math.random() > 0.5 ? '0' : '1';
              ctx!.fillStyle = `rgba(0, 3, 255, ${lensStrength})`;
            } else {
              ctx!.fillStyle = `rgba(0, 3, 255, ${alpha * 0.6})`;
            }

            const shiftX = (dx / dist) * 10 * lensStrength;
            const shiftY = (dy / dist) * 10 * lensStrength;

            ctx!.fillText(
              char,
              posX + CHAR_SIZE / 2 - shiftX,
              posY + CHAR_SIZE / 2 - shiftY,
            );
          } else if (char) {
            ctx!.fillStyle = `rgba(0, 3, 255, ${alpha * 0.7})`;
            ctx!.fillText(char, posX + CHAR_SIZE / 2, posY + CHAR_SIZE / 2);
          }
        }
      }

      time += 0.01;
      animId = requestAnimationFrame(render);
    }

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas!.getBoundingClientRect();
      mousePos.x = e.clientX - rect.left;
      mousePos.y = e.clientY - rect.top;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('resize', resize);

    resize();
    render();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '66vh',
        zIndex: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
      }}
    />
  );
}
