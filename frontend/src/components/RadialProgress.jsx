import { motion } from 'motion/react';

/* Radial progress ring (ReUI "progress → radial" style): track + animated arc + centred value. */
export function RadialProgress({ progress = 0, size = 96, strokeWidth = 6, label }) {
  const r = (size - strokeWidth) / 2;
  const c = 2 * Math.PI * r;
  const done = progress >= 100;

  return (
    <div className="relative inline-flex flex-col items-center gap-2" role="progressbar"
         aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none"
                  stroke="rgba(239,230,211,.1)" strokeWidth={strokeWidth} />
          <motion.circle
            cx={size / 2} cy={size / 2} r={r} fill="none" strokeLinecap="round"
            stroke={done ? 'var(--color-verdigris)' : 'var(--color-ember)'}
            strokeWidth={strokeWidth} strokeDasharray={c}
            animate={{ strokeDashoffset: c * (1 - progress / 100) }}
            transition={{ ease: 'easeOut', duration: 0.25 }}
            style={{ filter: `drop-shadow(0 0 6px ${done ? 'rgba(79,209,176,.6)' : 'rgba(255,90,31,.6)'})` }}
          />
        </svg>
        <span className="absolute inset-0 grid place-items-center font-mono text-sm tabular-nums text-bone">
          {Math.round(progress)}<span className="text-ash text-[10px] ml-0.5">%</span>
        </span>
      </div>
      {label && <span className="font-mono text-[11px] uppercase tracking-[.18em] text-ash">{label}</span>}
    </div>
  );
}
