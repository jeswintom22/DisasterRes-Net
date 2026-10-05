import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { motion, useScroll, useSpring, useTransform, AnimatePresence } from 'motion/react';
import Lenis from 'lenis';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { ThinkingOrbs } from './components/ThinkingOrbs';
import { RadialProgress } from './components/RadialProgress';
import { Dock } from './components/Dock';
import { Text3DFlip } from './components/Text3DFlip';
import { AIPromptBox } from './components/AIPromptBox';

const HERO_CLASS =
  'text-[clamp(3.5rem,13vw,10.5rem)] font-extrabold leading-[.88] tracking-[-0.045em] text-bone whitespace-nowrap';
const SHRINK_RANGE = 320; // px of scroll over which the hero travels to the corner
const END_SCALE = 0.17;
const END_X = 24;
const END_Y = 12;

/** Pick an orb look from the backend's progress message. */
function orbFor(message = '') {
  const m = message.toLowerCase();
  if (/search|fetch|imager|download|scene|geocod|locat/.test(m))
    return { label: 'Searching', state: 'searching', variant: 'lighthouse' };
  if (/reason|narrat|report|write|summar|compos|llm|consensus/.test(m))
    return { label: 'Reasoning', state: 'reasoning', variant: 'twins' };
  if (/cache|retry/.test(m)) return { label: 'Retrying', state: 'retrying', variant: 'surge' };
  return { label: 'Working', state: 'working', variant: 'gyro' };
}

/** Reveal `text` token by token (words + whitespace), like an LLM stream. */
function useTokenStream(text, enabled) {
  const tokens = useMemo(() => text.match(/\S+\s*/g) || [], [text]);
  const [shown, setShown] = useState(enabled ? 0 : Infinity);
  useEffect(() => {
    if (!enabled) return;
    const total = tokens.length;
    // ~28ms per token, but never longer than 9s or shorter than 1.2s overall.
    const duration = Math.min(9000, Math.max(1200, total * 28));
    const t0 = performance.now();
    let raf;
    const step = (now) => {
      const p = Math.min(1, (now - t0) / duration);
      setShown(Math.ceil(p * total));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [tokens, enabled]);
  const visible = enabled ? tokens.slice(0, shown).join('') : text;
  return { visible, streaming: enabled && shown < tokens.length };
}

function Report({ msg, onImageClick }) {
  const { visible, streaming } = useTokenStream(msg.content || '', !!msg.stream);
  const html = DOMPurify.sanitize(marked.parse(visible));
  const f = msg.facts || {};
  return (
    <div className="max-w-[88%] border-l-[3px] border-ember/80 bg-basalt-2/40 backdrop-blur-sm p-5 rounded-r-2xl shadow-[0_4px_24px_-8px_rgba(0,0,0,0.5)]">
      <div className="md prose prose-invert prose-p:leading-relaxed prose-pre:bg-basalt-3" dangerouslySetInnerHTML={{ __html: html }} />
      {streaming && <span className="caret inline-block w-2 h-4 bg-ember ml-1 align-middle animate-pulse" aria-hidden />}
      {msg.meta && !streaming && (
        <div className="mt-5 border-t border-bone/10 pt-4 flex flex-wrap gap-x-6 gap-y-2 font-mono text-[11px] uppercase tracking-[.18em] text-ash/80">
          <span className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-ember"></span> Mode · {msg.meta.narration || 'deterministic'}</span>
          <span className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-bone"></span> Analysed · {f.images_analysed ?? 0}</span>
          <span className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-bone/50"></span> Rejected · {f.images_rejected ?? 0}</span>
        </div>
      )}
      {f.visualizations && !streaming && (
        <div className="mt-4 grid grid-cols-2 gap-2">
           {Object.entries(f.visualizations).map(([name, b64]) => (
             <div key={name} className="relative group cursor-pointer" onClick={() => onImageClick?.(b64)}>
               <img src={b64} alt={name} className="w-full h-32 object-cover rounded-lg border border-bone/10 hover:border-ember transition-colors" />
               <div className="absolute bottom-1 left-1 bg-black/60 px-2 py-0.5 rounded text-[10px] text-bone font-mono uppercase backdrop-blur-md">{name}</div>
             </div>
           ))}
        </div>
      )}
      {msg.result_image && !streaming && !f.visualizations && (
        <div className="mt-6 -mx-2 -mb-2 overflow-hidden rounded-xl border border-bone/10 cursor-pointer" onClick={() => onImageClick?.(msg.result_image)}>
          <img src={msg.result_image} alt="Result" className="w-full h-auto object-cover hover:scale-[1.02] transition-transform duration-500" />
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [stage, setStage] = useState(null);
  const [upload, setUpload] = useState(0);
  const [pendingUpload, setPendingUpload] = useState(null);
  const [lightbox, setLightbox] = useState(null);
  const [start, setStart] = useState({ x: 24, y: 120 });
  const placeholder = useRef(null);
  const endRef = useRef(null);
  const eventSourceRef = useRef(null);
  
  const hasChat = messages.length > 0 || stage !== null || pendingUpload !== null;

  /* ── scroll-linked hero: shrinks and flies to the top-left corner ── */
  const { scrollY: rawScroll } = useScroll();
  // Spring on top of Lenis' inertia, so the logo glides instead of tracking the wheel 1:1.
  const scrollY = useSpring(rawScroll, { stiffness: 140, damping: 28, mass: 0.35 });
  const x = useTransform(scrollY, [0, SHRINK_RANGE], [start.x, END_X]);
  const y = useTransform(scrollY, [0, SHRINK_RANGE], [start.y, END_Y]);
  const scale = useTransform(scrollY, [0, SHRINK_RANGE], [1, END_SCALE]);
  const fade = useTransform(scrollY, [0, SHRINK_RANGE * 0.5], [1, 0]);
  const bar = useTransform(scrollY, [SHRINK_RANGE * 0.6, SHRINK_RANGE], [0, 1]);

  // Inertial smooth scrolling (Lenis).
  const lenis = useRef(null);
  useEffect(() => {
    const l = new Lenis({ lerp: 0.085, wheelMultiplier: 0.9 });
    lenis.current = l;
    let raf;
    const loop = (t) => { l.raf(t); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); l.destroy(); lenis.current = null; };
  }, []);

  // Measure where the (invisible) in-flow hero sits so the fixed one starts exactly on top of it.
  useLayoutEffect(() => {
    const measure = () => {
      const r = placeholder.current?.getBoundingClientRect();
      if (r) setStart({ x: r.left, y: r.top + window.scrollY });
    };
    measure();
    window.addEventListener('resize', measure);
    document.fonts?.ready.then(measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  /* ── pin to bottom while chatting; releases if the user scrolls up, re-pins near the bottom ── */
  const pinned = useRef(false);
  const scrollBottom = () => {
    const top = document.documentElement.scrollHeight;
    if (lenis.current) lenis.current.scrollTo(top, { immediate: true, force: true });
    else window.scrollTo({ top, behavior: 'auto' });
  };
  const toBottom = () => { pinned.current = true; scrollBottom(); };

  useEffect(() => {
    const gap = () => document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
    const onScroll = () => { if (gap() < 140) pinned.current = true; };
    const onUserUp = (e) => { if (e.deltaY < 0 || e.type === 'touchmove') { if (gap() > 140) pinned.current = false; } };
    const ro = new ResizeObserver(() => { if (pinned.current) scrollBottom(); });
    ro.observe(document.body);
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('wheel', onUserUp, { passive: true });
    window.addEventListener('touchmove', onUserUp, { passive: true });
    return () => {
      ro.disconnect();
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('wheel', onUserUp);
      window.removeEventListener('touchmove', onUserUp);
    };
  }, []);

  // Any new message or status line while pinned keeps the view at the bottom.
  useEffect(() => { if (pinned.current) scrollBottom(); }, [messages, stage]);

  /* ── real backend: POST /api/chat → SSE /api/chat/stream/:job ── */
  const handleSend = async (query, imageUrl = null, base64 = null) => {
    setMessages((m) => [...m, { role: 'user', content: query, image: imageUrl }]);
    setStage({ label: 'Working', state: 'working', variant: 'gyro', detail: 'starting investigation' });
    toBottom();
    const lastImage = imageUrl || [...messages].reverse().find(m => m.role === 'user' && m.image)?.image;
    let payload = { query };
    if (base64) payload.image = base64;
    const fail = (text) => {
      setStage(null);
      setMessages((m) => [...m, { role: 'error', content: text }]);
      toBottom();
    };
    try {
      const res = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Could not start chat');
      const source = new EventSource('/api/chat/stream/' + data.job_id);
      eventSourceRef.current = source;
      source.onmessage = (e) => {
        const item = JSON.parse(e.data);
        if (item.stage === 'complete') {
          source.close();
          setStage(null);
          const r = item.result || {};
          setMessages((m) => [...m, { role: 'ai', stream: true, content: r.response || '', facts: r.facts, meta: { narration: r.narration }, result_image: lastImage }]);
          toBottom();
        } else if (item.stage === 'error') {
          source.close();
          fail(item.message || 'The investigation failed.');
        } else {
          setStage({ ...orbFor(item.message), detail: item.message });
        }
      };
      source.onerror = () => { source.close(); fail('Lost connection to the backend. Is Flask running on :5000?'); };
    } catch (err) {
      fail(err.message);
    }
  };

  const handleStop = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setStage(null);
  };

  const handleUpload = async (file) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = async () => {
      const base64 = reader.result;
      for (let i = 4; i <= 100; i += 12) {
        setUpload(Math.min(i, 100));
        await new Promise(r => setTimeout(r, 90));
      }
      setUpload(100);
      await new Promise(r => setTimeout(r, 600));
      setUpload(0);
      setPendingUpload({ url, base64 });
    };
  };

  useEffect(() => { document.title = 'DisasterRes · Satellite damage assessment'; }, []);

  return (
    <div className="relative min-h-[150vh] overflow-x-hidden pb-52">
      <div className="atmosphere" /><div className="topo" /><div className="grain" />

      {/* top bar fades in once the hero has docked */}
      <motion.div style={{ opacity: bar }}
        className="pointer-events-none fixed inset-x-0 top-0 z-30 h-14 border-b border-bone/10 bg-basalt/70 backdrop-blur-xl" />
      <div className="fixed right-6 top-4 z-40 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[.22em] text-ash">
        <span className="h-2 w-2 animate-pulse rounded-full bg-ember shadow-[0_0_10px_var(--color-ember)]" />
        Live · Sweep desk 01
      </div>

      {/* the hero: fixed, driven by scroll */}
      <motion.div style={{ x, y, scale, transformOrigin: '0 0' }} className="fixed left-0 top-0 z-50">
        <Text3DFlip as="h1" className={HERO_CLASS} flipTextClassName="text-ember" staggerDuration={0.035}>
          DisasterRes
        </Text3DFlip>
      </motion.div>

      <main className="relative z-10 mx-auto max-w-5xl px-6">
        {/* in-flow spacer for the hero; its text fades out as you scroll */}
        <section className="pt-24">
          <motion.p style={{ opacity: fade }} className="font-serif text-2xl italic text-ember-soft">
            from orbit, to answers in minutes —
          </motion.p>
          <h2 ref={placeholder} aria-hidden className={`${HERO_CLASS} invisible`}>DisasterRes</h2>
          <motion.p style={{ opacity: fade }} className="mt-6 max-w-xl pb-10 text-lg leading-relaxed text-ash">
            Multi-vendor satellite damage assessment. Name a place and a crew of agents
            sweeps the imagery, argues over it, and files the report.
          </motion.p>
        </section>

        <motion.div layout transition={{ type: 'spring', stiffness: 120, damping: 22 }} className={hasChat ? "fixed inset-x-0 bottom-8 z-40 px-6" : "relative mt-2 mb-12 z-40"}>
          <div className="mx-auto max-w-3xl">
            <AIPromptBox 
              onSubmit={(q) => {
                if (!q.trim() && !pendingUpload) return;
                handleSend(q || 'Analyze this imagery', pendingUpload?.url, pendingUpload?.base64);
                setPendingUpload(null);
              }} 
              onAttach={handleUpload} 
              isLoading={!!stage}
              thumbnail={pendingUpload?.url}
              onClearThumbnail={() => setPendingUpload(null)}
              onStop={handleStop}
            />
          </div>
        </motion.div>

        {/* transcript */}
        <section className="min-h-[70vh] space-y-8 pt-10">
          {messages.length === 0 && !stage && (
            <motion.p style={{ opacity: fade }} className="font-mono text-xs uppercase tracking-[.2em] text-ash/60">
              — awaiting coordinates —
            </motion.p>
          )}
          {messages.map((msg, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'user' && (
                <div className="flex flex-col items-end gap-2">
                  {msg.image && (
                    <img src={msg.image} alt="Uploaded" className="max-w-[240px] rounded-xl border border-bone/10 cursor-pointer hover:opacity-90 transition-opacity" onClick={() => setLightbox(msg.image)} />
                  )}
                  {msg.content && (
                    <div className="max-w-[80%] rounded-2xl rounded-br-[4px] bg-bone/90 backdrop-blur pl-4 pr-6 py-2.5 text-[15px] leading-snug text-basalt shadow-sm text-left">{msg.content}</div>
                  )}
                </div>
              )}
              {msg.role === 'ai' && <Report msg={msg} onImageClick={setLightbox} />}
              {msg.role === 'error' && (
                <div className="max-w-[80%] rounded-2xl bg-ember/10 px-5 py-3 font-mono text-sm text-ember-soft ring-1 ring-ember/40">{msg.content}</div>
              )}
            </motion.div>
          ))}
          {stage && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} key={stage.label}>
              <ThinkingOrbs {...stage} />
            </motion.div>
          )}
          <div ref={endRef} />
        </section>
      </main>

      {upload > 0 && (
        <motion.div initial={{ opacity: 0, scale: .8 }} animate={{ opacity: 1, scale: 1 }}
          className="fixed right-8 top-24 z-40 rounded-3xl bg-basalt-2/90 p-5 backdrop-blur-xl hairline">
          <RadialProgress progress={upload} label={upload >= 100 ? 'Uploaded' : 'Uploading'} />
        </motion.div>
      )}

      <Dock onSelect={(id) => id === 'upload' && document.querySelector('input[type="file"]')?.click()} />

      <AnimatePresence>
        {lightbox && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] grid place-items-center bg-black/80 p-6 backdrop-blur-xl"
            onClick={() => setLightbox(null)}
          >
            <button className="absolute top-6 right-6 text-bone hover:text-white bg-white/10 hover:bg-white/20 rounded-full p-2 transition" onClick={() => setLightbox(null)}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
            <motion.img 
              initial={{ scale: 0.95 }} animate={{ scale: 1 }} exit={{ scale: 0.95 }}
              src={lightbox} alt="Enlarged" className="max-h-[90vh] max-w-[90vw] rounded-lg object-contain shadow-2xl" 
              onClick={e => e.stopPropagation()} 
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
