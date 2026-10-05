import { useState } from 'react';
import { motion } from 'motion/react';
import { MessageSquare, UploadCloud, Satellite, History } from 'lucide-react';

const ITEMS = [
  { id: 'chat', label: 'Select chat', Icon: MessageSquare },
  { id: 'upload', label: 'Upload file', Icon: UploadCloud },
  { id: 'sweep', label: 'Satellite sweep', Icon: Satellite },
  { id: 'history', label: 'Run history', Icon: History },
];

/* Sleek vertical sidebar dock on the left, polished and compact. */
export function Dock({ onSelect, defaultActive = 'chat' }) {
  const [active, setActive] = useState(defaultActive);
  const [hover, setHover] = useState(null);

  return (
    <div className="fixed left-5 top-1/2 -translate-y-1/2 z-50">
      <motion.nav
        initial={{ x: -80, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 180, damping: 20, delay: .6 }}
        className="flex flex-col items-center gap-3 rounded-full px-2.5 py-4 backdrop-blur-2xl bg-basalt-3/40 border border-bone/5 shadow-2xl"
      >
        {ITEMS.map(({ id, label, Icon }) => {
          const isActive = active === id;
          return (
            <div key={id} className="relative flex items-center justify-center">
              {hover === id && (
                <motion.span initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                  className="absolute left-[3.5rem] whitespace-nowrap rounded-md bg-bone/90 backdrop-blur-md px-2.5 py-1 font-mono text-[11px] text-basalt uppercase tracking-widest shadow-xl">
                  {label}
                </motion.span>
              )}
              <motion.button
                aria-label={label}
                onHoverStart={() => setHover(id)} onHoverEnd={() => setHover(null)}
                onClick={() => { setActive(id); onSelect?.(id); }}
                whileHover={{ scale: 1.15 }}
                whileTap={{ scale: .92 }}
                className={`grid h-[42px] w-[42px] place-items-center rounded-[14px] transition-colors duration-300
                  ${isActive ? 'bg-ember text-basalt shadow-lg shadow-ember/20' : 'bg-transparent text-ash hover:bg-bone/10 hover:text-bone'}`}
              >
                <Icon size={18} strokeWidth={isActive ? 2 : 1.75} />
              </motion.button>
            </div>
          );
        })}
      </motion.nav>
    </div>
  );
}
