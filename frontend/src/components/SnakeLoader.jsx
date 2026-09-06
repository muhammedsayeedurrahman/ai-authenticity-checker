import React, { useCallback, useEffect, useRef, useState } from 'react';

/** Joins truthy class names — local stand-in for shadcn's `cn` (no such helper exists in this codebase). */
function cn(...classes) {
  return classes.filter(Boolean).join(' ');
}

/* -------------------------------------------------------------------------- */
/*                              frame generation                              */
/* -------------------------------------------------------------------------- */

/** Neighbours of a cell on a `w x w` grid, as flat indices. */
function neighbours(p, w) {
  const row = (p / w) | 0;
  const col = p % w;
  const out = [];
  if (row > 0) out.push(p - w);
  if (row < w - 1) out.push(p + w);
  if (col > 0) out.push(p - 1);
  if (col < w - 1) out.push(p + 1);
  return out;
}

/** Shortest path length from `from` to `to`, walls excluded. */
function distance(from, to, walls, w, n) {
  if (from === to) return 0;
  const seen = new Set([from]);
  let front = [from];
  let d = 0;
  while (front.length) {
    d++;
    const next = [];
    for (const cell of front) {
      for (const nb of neighbours(cell, w)) {
        if (nb === to) return d;
        if (!seen.has(nb) && !walls.has(nb)) {
          seen.add(nb);
          next.push(nb);
        }
      }
    }
    front = next;
  }
  return n + 1;
}

/** Size of the free region reachable from `start`. */
function reachable(start, walls, w) {
  const seen = new Set([start]);
  const queue = [start];
  let i = 0;
  while (i < queue.length) {
    for (const nb of neighbours(queue[i++], w)) {
      if (!seen.has(nb) && !walls.has(nb)) {
        seen.add(nb);
        queue.push(nb);
      }
    }
  }
  return seen.size;
}

function firstFree(walls, n) {
  for (let i = 0; i < n; i++) if (!walls.has(i)) return i;
  return -1;
}

/**
 * True when moving into `cell` leaves the snake room to keep going: the open
 * region it lands in must hold at least its own body.
 */
function isSafe(cell, apple, tail, used, length, w) {
  const walls = new Set(used);
  if (cell !== apple) walls.delete(tail);
  walls.add(cell);
  return reachable(cell, walls, w) > length;
}

/**
 * Plays a full game of snake and returns it as animation frames.
 *
 * The snake takes the shortest path to each apple, but only steps where the
 * open space left in front of it is bigger than its own body — so it does not
 * box itself in. The game ends when it runs out of moves or out of steps.
 * Every call plays a different game.
 */
export function generateSnakeFrames(width = 7) {
  const w = Math.max(3, Math.floor(width));
  const n = w * w;

  const snake = [0, 1];
  const used = new Set(snake);
  const body = [];
  const apples = [];

  const spawn = () => {
    const free = [];
    for (let i = 0; i < n; i++) if (!used.has(i)) free.push(i);
    return free.length ? free[(Math.random() * free.length) | 0] : -1;
  };

  let apple = spawn();
  // Keeps every loop a similar length, however lucky the snake gets.
  let budget = n * 5;

  while (snake.length < n && apple >= 0 && --budget > 0) {
    body.push([...snake]);
    apples.push([apple]);

    const head = snake[snake.length - 1];
    const tail = snake[0];

    // The tail square is legal: it empties on the same step the head enters.
    const open = neighbours(head, w).filter((nb) => !used.has(nb) || nb === tail);

    let move = -1;
    let best = n + 1;
    for (const nb of open) {
      if (!isSafe(nb, apple, tail, used, snake.length, w)) continue;
      const d = distance(nb, apple, used, w, n);
      if (d < best) {
        best = d;
        move = nb;
      }
    }

    // No safe step towards the apple — head for the roomiest square instead.
    if (move < 0) {
      let room = -1;
      for (const nb of open) {
        const sim = new Set(used);
        sim.delete(tail);
        sim.add(nb);
        const seed = firstFree(sim, n);
        const size = seed >= 0 ? reachable(seed, sim, w) : 0;
        if (size > room) {
          room = size;
          move = nb;
        }
      }
      if (move < 0) break;
    }

    const grows = move === apple;
    if (!grows) used.delete(snake.shift());
    snake.push(move);
    used.add(move);
    if (grows) apple = spawn();
  }

  // Two blinks of the finished board close the loop.
  const full = [...snake];
  body.push(full, full, [], full, []);
  apples.push([], [], [], [], []);

  return { body, apples };
}

/* -------------------------------------------------------------------------- */
/*                                  component                                 */
/* -------------------------------------------------------------------------- */

export function SnakeLoader({
  width = 7,
  speed = 80,
  playing = true,
  loop = true,
  onComplete,
  snakeColor = 'currentColor',
  appleColor = '#A3E635',
  dotClassName,
  className,
  style,
  ...props
}) {
  const gridRef = useRef(null);
  const frame = useRef(0);
  const timer = useRef(null);

  const [round, setRound] = useState(0);
  const [game, setGame] = useState(null);

  // Held in a ref so an inline callback does not restart the animation.
  const completeRef = useRef(onComplete);
  completeRef.current = onComplete;

  useEffect(() => {
    setGame(generateSnakeFrames(width));
    frame.current = 0;
  }, [width, round]);

  const paint = useCallback(
    (dots, index) => {
      if (!game) return;
      const body = game.body[index];
      if (!body) return;
      const apple = game.apples[index];

      dots.forEach((dot, i) => {
        const isApple = apple?.includes(i) ?? false;
        dot.classList.toggle('active', !isApple && body.includes(i));
        dot.classList.toggle('accent', isApple);
      });
    },
    [game],
  );

  useEffect(() => {
    if (!game || !playing) return;

    const grid = gridRef.current;
    if (!grid) return;
    const dots = Array.from(grid.children);

    if (frame.current >= game.body.length) frame.current = 0;

    timer.current = setInterval(() => {
      paint(dots, frame.current);

      if (frame.current + 1 >= game.body.length) {
        completeRef.current?.();
        if (loop) {
          setRound((r) => r + 1);
          return;
        }
        clearInterval(timer.current);
      }
      frame.current++;
    }, speed);

    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [game, playing, speed, loop, paint]);

  return (
    <div
      {...props}
      ref={gridRef}
      role="status"
      aria-label="Loading"
      className={cn('grid w-fit gap-0.5', className)}
      style={{
        gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))`,
        '--snake-color': snakeColor,
        '--apple-color': appleColor,
        ...style,
      }}
    >
      {Array.from({ length: width * width }).map((_, i) => (
        <div
          key={i}
          className={cn(
            'size-1.5 rounded-[1px] transition-colors duration-100',
            'bg-[color-mix(in_srgb,currentColor_12%,transparent)]',
            '[&.active]:bg-[var(--snake-color)] [&.accent]:bg-[var(--apple-color)]',
            dotClassName,
          )}
        />
      ))}
    </div>
  );
}

export default SnakeLoader;
