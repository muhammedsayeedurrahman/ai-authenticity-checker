import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

const PARTICLE_COUNT = 90;
const THRESHOLD = 80;

export default function NeuralBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedMotion) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
    camera.position.z = 300;

    // Particles
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const velocities = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 500;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 500;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 200;
      velocities.push({
        x: (Math.random() - 0.5) * 0.3,
        y: (Math.random() - 0.5) * 0.3,
        z: (Math.random() - 0.5) * 0.1,
      });
    }

    const pointsGeo = new THREE.BufferGeometry();
    pointsGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const pointsMat = new THREE.PointsMaterial({
      color: 0x6366f1,
      size: 2.5,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    scene.add(new THREE.Points(pointsGeo, pointsMat));

    const lineMat = new THREE.LineBasicMaterial({
      color: 0x3f3f46,
      transparent: true,
      opacity: 0.15,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const mouse = { x: 0, y: 0 };
    const onMouseMove = (e) => {
      mouse.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouse.y = -(e.clientY / window.innerHeight - 0.5) * 2;
    };
    document.addEventListener('mousemove', onMouseMove);

    let frameCount = 0;
    let animId;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      frameCount++;
      if (frameCount % 2 !== 0) return;

      const pos = pointsGeo.attributes.position.array;
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        pos[i * 3] += velocities[i].x + mouse.x * 0.05;
        pos[i * 3 + 1] += velocities[i].y + mouse.y * 0.05;
        pos[i * 3 + 2] += velocities[i].z;
        if (pos[i * 3] > 250) pos[i * 3] = -250;
        if (pos[i * 3] < -250) pos[i * 3] = 250;
        if (pos[i * 3 + 1] > 250) pos[i * 3 + 1] = -250;
        if (pos[i * 3 + 1] < -250) pos[i * 3 + 1] = 250;
      }
      pointsGeo.attributes.position.needsUpdate = true;

      if (frameCount % 6 === 0) {
        // Remove old line segments
        for (let i = scene.children.length - 1; i >= 0; i--) {
          if (scene.children[i].isLineSegments) {
            scene.children[i].geometry.dispose();
            scene.remove(scene.children[i]);
          }
        }

        const lp = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
          for (let j = i + 1; j < PARTICLE_COUNT; j++) {
            const dx = pos[i * 3] - pos[j * 3];
            const dy = pos[i * 3 + 1] - pos[j * 3 + 1];
            const dz = pos[i * 3 + 2] - pos[j * 3 + 2];
            if (Math.sqrt(dx * dx + dy * dy + dz * dz) < THRESHOLD) {
              lp.push(
                pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2],
                pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2],
              );
            }
          }
        }
        if (lp.length > 0) {
          const lg = new THREE.BufferGeometry();
          lg.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
          scene.add(new THREE.LineSegments(lg, lineMat));
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(animId);
      document.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      pointsGeo.dispose();
      pointsMat.dispose();
      lineMat.dispose();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 0,
        pointerEvents: 'none',
        opacity: 0.18,
      }}
    />
  );
}
