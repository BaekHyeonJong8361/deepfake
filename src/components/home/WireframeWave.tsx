import { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function WireframeWave() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / container.clientHeight,
      0.1,
      1000,
    );
    camera.position.z = 20;
    camera.position.y = 9;
    camera.position.x = -4;
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const color = new THREE.Color(0x0003ff);
    const geometry = new THREE.PlaneGeometry(40, 40, 32, 32);

    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0.0 },
        color: { value: color },
        amplitude: { value: 2.0 },
        frequency: { value: 0.2 },
      },
      vertexShader: `
        uniform float time;
        uniform float amplitude;
        uniform float frequency;
        varying vec2 vUv;

        float noise(vec2 p) {
          return sin(p.x * frequency + time) * cos(p.y * frequency + time) * amplitude;
        }

        void main() {
          vUv = uv;
          vec3 pos = position;
          pos.z += noise(pos.xy);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 color;
        void main() {
          gl_FragColor = vec4(color, 0.8);
        }
      `,
      wireframe: true,
      transparent: true,
    });

    const plane = new THREE.Mesh(geometry, material);
    plane.rotation.x = -Math.PI / 2;
    scene.add(plane);

    const clock = new THREE.Clock();
    let animId: number;

    function animate() {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();
      material.uniforms.time.value = elapsed;
      plane.rotation.z = elapsed * 0.05;
      renderer.render(scene, camera);
    }
    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
}
