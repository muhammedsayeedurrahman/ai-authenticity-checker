import { useEffect } from 'react';
import { motion, useMotionValue, useTransform, animate } from 'framer-motion';

/**
 * Count-up number display. Animates from its previous value to `value`
 * whenever it changes (including the initial 0 -> value mount animation).
 *
 * Renders a transformed MotionValue directly as a motion.span child —
 * framer-motion's documented pattern for animated counters, updates the
 * DOM text node directly without a React re-render per tick.
 */
export default function AnimatedNumber({ value, decimals = 0, suffix = '', duration = 0.8 }) {
  const motionValue = useMotionValue(0);
  const display = useTransform(motionValue, (v) => `${v.toFixed(decimals)}${suffix}`);

  useEffect(() => {
    const controls = animate(motionValue, Number(value) || 0, {
      duration,
      ease: [0.22, 1, 0.36, 1],
    });
    return controls.stop;
  }, [value, duration, motionValue]);

  return <motion.span>{display}</motion.span>;
}
