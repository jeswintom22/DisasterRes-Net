import { useRef, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowUp, Globe, BrainCog, PenLine, Paperclip, Square, X } from 'lucide-react';

/* AI prompt box: auto-growing textarea, Search / Think / Canvas mode toggles, attach, morphing send button. */
const MODES = [
  { id: 'search', label: 'Search', Icon: Globe },
  { id: 'think', label: 'Think', Icon: BrainCog },
  { id: 'canvas', label: 'Canvas', Icon: PenLine },
];

export function AIPromptBox({ onSubmit, onAttach, isLoading, thumbnail, onClearThumbnail, onStop }) {
  const [value, setValue] = useState('');
  const [modes, setModes] = useState({});
  const [focused, setFocused] = useState(false);
  const ref = useRef(null);
  const fileRef = useRef(null);

  const thumbnails = Array.isArray(thumbnail) ? thumbnail : (thumbnail ? [thumbnail] : []);

  useEffect(() => {
    const el = ref.current; if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [value]);

  const send = () => {
    if ((!value.trim() && thumbnails.length === 0) || isLoading) return;
    onSubmit?.(value.trim(), Object.keys(modes).filter(k => modes[k]));
    setValue('');
  };

  return (
    <motion.div
      animate={{ boxShadow: focused
        ? '0 0 0 1px rgba(255,138,76,.6), 0 20px 60px -20px rgba(255,90,31,.45)'
        : '0 0 0 1px rgba(239,230,211,.1), 0 20px 50px -25px rgba(0,0,0,.8)' }}
      className="rounded-[28px] bg-basalt-2/90 p-3 backdrop-blur-xl"
    >
      <AnimatePresence>
        {thumbnails.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex flex-wrap items-center gap-2 mb-3 px-2 pt-1"
          >
            {thumbnails.map((item, idx) => {
              const url = typeof item === 'string' ? item : item?.url;
              return (
                <div key={idx} className="relative group/thumb">
                  <img
                    src={url}
                    alt={`Upload ${idx + 1}`}
                    className="h-16 w-16 sm:h-20 sm:w-20 object-cover rounded-xl border border-bone/20 shadow-md group-hover/thumb:border-ember/60 transition-colors"
                  />
                  {onClearThumbnail && (
                    <button
                      type="button"
                      aria-label="Remove image"
                      onClick={() => onClearThumbnail(idx)}
                      className="absolute -top-1.5 -right-1.5 bg-basalt-3/90 text-ash hover:text-bone hover:bg-ember rounded-full p-1 border border-bone/20 shadow-lg transition-colors"
                    >
                      <X size={12} strokeWidth={2.5} />
                    </button>
                  )}
                </div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
      <textarea
        ref={ref} rows={1} value={value} placeholder="Name a place, upload imagery, or ask for a damage report…"
        onChange={e => setValue(e.target.value)}
        onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
        className="scrollbar-thin w-full resize-none bg-transparent px-3 pt-2 pb-1 text-[16px] text-bone
                   placeholder:text-ash/70 placeholder:font-serif placeholder:italic placeholder:text-[17px] focus:outline-none"
      />
      <input
        type="file"
        ref={fileRef}
        multiple
        className="hidden"
        accept="image/*"
        onChange={e => {
          if (e.target.files && e.target.files.length > 0) {
            onAttach?.(Array.from(e.target.files));
            e.target.value = '';
          }
        }}
      />
      <div className="mt-1 flex items-center gap-1.5">
        <button type="button" aria-label="Attach file" onClick={() => fileRef.current?.click()}
          className="grid h-9 w-9 place-items-center rounded-full text-ash transition hover:bg-basalt-3 hover:text-bone">
          <Paperclip size={17} />
        </button>
        <span className="mx-1 h-5 w-px bg-bone/10" />
        {MODES.map(({ id, label, Icon }) => {
          const on = !!modes[id];
          return (
            <button key={id} type="button" onClick={() => setModes(m => ({ ...m, [id]: !m[id] }))}
              className={`flex h-9 items-center gap-1.5 rounded-full px-3 font-mono text-[11px] uppercase tracking-wider transition
                ${on ? 'bg-ember/15 text-ember-soft ring-1 ring-ember/50' : 'text-ash hover:bg-basalt-3 hover:text-bone'}`}>
              <Icon size={14} />
              <AnimatePresence initial={false}>
                {(on || focused) && (
                  <motion.span initial={{ width: 0, opacity: 0 }} animate={{ width: 'auto', opacity: 1 }}
                    exit={{ width: 0, opacity: 0 }} className="overflow-hidden">{label}</motion.span>
                )}
              </AnimatePresence>
            </button>
          );
        })}
        <motion.button
          type="button" aria-label={isLoading ? 'Stop' : 'Send'} onClick={isLoading ? onStop : send}
          whileTap={{ scale: .9 }} whileHover={{ scale: 1.06 }}
          className={`ml-auto grid h-10 w-10 place-items-center rounded-full transition-colors
            ${isLoading ? 'bg-ash/20 text-ash hover:bg-ember hover:text-basalt' : (value.trim() || thumbnails.length > 0) ? 'bg-ember text-basalt shadow-[0_0_24px_rgba(255,90,31,.55)]' : 'bg-basalt-3 text-ash'}`}
        >
          {isLoading ? <Square size={14} fill="currentColor" /> : <ArrowUp size={18} strokeWidth={2.4} />}
        </motion.button>
      </div>
    </motion.div>
  );
}
