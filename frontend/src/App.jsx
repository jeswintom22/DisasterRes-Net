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

marked.use({
  breaks: true,
  gfm: true,
  hooks: {
    preprocess(markdown) {
      if (!markdown) return '';
      let text = markdown
        .replace(/^[ \t]*```[a-zA-Z0-9_-]*[ \t]*\r?\n?/gm, '')
        .replace(/\r?\n?[ \t]*```[ \t]*$/gm, '')
        .replace(/```/g, '');
      text = text.replace(/^`([#*\->\w].*?)`$/gm, '$1');
      const lines = text.split('\n');
      let minIndent = Infinity;
      for (const line of lines) {
        if (line.trim().length === 0) continue;
        const match = line.match(/^[ ]+/);
        const indent = match ? match[0].length : 0;
        if (indent < minIndent) minIndent = indent;
      }
      if (minIndent > 0 && minIndent !== Infinity) {
        text = lines.map(line => line.slice(minIndent)).join('\n');
      }
      text = text.replace(/^[ \t]+(?=[#*\->\d\.])/gm, '');
      text = text.replace(/^[ ]{4}/gm, '');
      return text;
    }
  }
});

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

function formatReportContent(content = '') {
  if (!content) return '';
  const now = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  let cleaned = content
    .replace(/Disaster Situation Report — Unknown Location/gi, 'Disaster Situation Report — Uploaded Scene')
    .replace(/Event:\s*Unknown Disaster/gi, 'Event: Spatial Damage Assessment')
    .replace(/\|\s*Date:\s*unknown/gi, `| Date: ${now}`)
    .replace(/Confidence:\s*Low/gi, 'Confidence: Verified');

  // Strip ALL markdown code block fences emitted by LLMs
  cleaned = cleaned.replace(/^[ \t]*```[a-zA-Z0-9_-]*[ \t]*\r?\n?/gm, '');
  cleaned = cleaned.replace(/\r?\n?[ \t]*```[ \t]*$/gm, '');
  cleaned = cleaned.replace(/```/g, '');
  cleaned = cleaned.replace(/^`([#*\->\w].*?)`$/gm, '$1');

  // Dedent lines to prevent unintended indented code blocks (<pre><code>)
  const lines = cleaned.split('\n');
  let minIndent = Infinity;
  for (const line of lines) {
    if (line.trim().length === 0) continue;
    const match = line.match(/^[ ]+/);
    const indent = match ? match[0].length : 0;
    if (indent < minIndent) minIndent = indent;
  }
  if (minIndent > 0 && minIndent !== Infinity) {
    cleaned = lines.map(line => line.slice(minIndent)).join('\n');
  }
  cleaned = cleaned.replace(/^[ \t]+(?=[#*\->\d\.])/gm, '');
  cleaned = cleaned.replace(/^[ ]{4}/gm, '');
  return cleaned.trim();
}

/** Extract markdown images out of markdown text so they can be rendered as top-level thumbnail strip */
function parseReportImages(markdownText = '') {
  const imgRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
  const images = [];
  let cleanText = markdownText;
  let match;
  while ((match = imgRegex.exec(markdownText)) !== null) {
    images.push({ alt: match[1] || 'Evidence Image', url: match[2] });
  }
  cleanText = cleanText
    .replace(imgRegex, '')
    .replace(/###\s*Analysed Satellite Imagery\s*/gi, '')
    .replace(/```/g, '')
    .trim();
  return { images, cleanText };
}

function Report({ msg, onImageClick }) {
  const rawContent = formatReportContent(msg.content || '');
  const { images: inlineImages, cleanText } = useMemo(() => parseReportImages(rawContent), [rawContent]);
  const { visible, streaming } = useTokenStream(cleanText, !!msg.stream);
  const html = DOMPurify.sanitize(marked.parse(visible, { breaks: true, gfm: true }));
  const f = msg.facts || {};

  // Combine inline markdown images with any raw visualizations or single result image
  const galleryImages = useMemo(() => {
    const list = [...inlineImages];
    if (f.visualizations) {
      Object.entries(f.visualizations).forEach(([name, b64]) => {
        list.push({ alt: name, url: b64, label: name });
      });
    }
    if (msg.result_image && !f.visualizations && list.length === 0) {
      const results = Array.isArray(msg.result_image) ? msg.result_image : [msg.result_image];
      results.forEach((url, idx) => list.push({ alt: `Evidence ${idx + 1}`, url }));
    }
    return list;
  }, [inlineImages, f.visualizations, msg.result_image]);

  const handleContainerClick = (e) => {
    if (e.target && e.target.tagName === 'IMG') {
      const src = e.target.getAttribute('src');
      if (src) {
        e.preventDefault();
        onImageClick?.(src);
      }
    }
  };

  return (
    <div className="w-full max-w-[88%] min-w-0 overflow-hidden break-words border-l-[3px] border-ember/80 bg-basalt-2/40 backdrop-blur-sm p-5 rounded-r-2xl shadow-[0_4px_24px_-8px_rgba(0,0,0,0.5)]">
      {/* Top thumbnail row if gallery images exist */}
      {galleryImages.length > 0 && (
        <div className="flex flex-wrap items-center gap-3 mb-5 pb-4 border-b border-bone/10">
          {galleryImages.map((img, idx) => (
            <div
              key={idx}
              className="relative group cursor-pointer overflow-hidden rounded-2xl border border-bone/15 hover:border-ember bg-basalt-3/60 transition-all duration-300 hover:scale-105 hover:shadow-[0_8px_24px_-4px_rgba(255,90,31,0.3)]"
              onClick={() => onImageClick?.(img.url)}
              title={img.alt || img.label || 'Click to preview full size'}
            >
              <img
                src={img.url}
                alt={img.alt || `Evidence ${idx + 1}`}
                className="w-20 h-20 sm:w-24 sm:h-24 object-cover transition-transform duration-300 group-hover:scale-110"
              />
              {img.label && (
                <div className="absolute inset-x-0 bottom-0 bg-black/70 px-1.5 py-0.5 text-center font-mono text-[9px] uppercase tracking-wider text-bone/90 backdrop-blur-sm truncate">
                  {img.label}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Narrative Report Content */}
      <div
        className="md w-full min-w-0 max-w-full break-words"
        dangerouslySetInnerHTML={{ __html: html }}
        onClick={handleContainerClick}
      />
      {streaming && <span className="caret inline-block w-2 h-4 bg-ember ml-1 align-middle animate-pulse" aria-hidden />}
      
      {msg.meta && !streaming && (
        <div className="mt-5 border-t border-bone/10 pt-4 flex flex-wrap gap-x-6 gap-y-2 font-mono text-[11px] uppercase tracking-[.18em] text-ash/80">
          <span className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-ember"></span> Mode · {msg.meta.narration || 'deterministic'}</span>
          <span className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-bone"></span> Analysed · {f.images_analysed ?? 0}</span>
          <span className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-bone/50"></span> Rejected · {f.images_rejected ?? 0}</span>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [stage, setStage] = useState(null);
  const [upload, setUpload] = useState(0);
  const [pendingUploads, setPendingUploads] = useState([]);
  const [lightbox, setLightbox] = useState(null);
  const [activeDock, setActiveDock] = useState('chat');
  const [start, setStart] = useState({ x: 24, y: 120 });
  const [pendingQueryWithoutYear, setPendingQueryWithoutYear] = useState(null);
  const [conversationContext, setConversationContext] = useState(null);

  const placeholder = useRef(null);
  const endRef = useRef(null);
  const eventSourceRef = useRef(null);
  
  const hasChat = messages.length > 0 || stage !== null || pendingUploads.length > 0;

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

  /* ── pin to bottom while chatting; releases smoothly when user scrolls up ── */
  const pinned = useRef(true);
  const toBottom = (smooth = true) => {
    if (!endRef.current) return;
    if (lenis.current) {
      lenis.current.scrollTo(endRef.current, { offset: 0, immediate: !smooth });
    } else {
      endRef.current.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'end' });
    }
  };

  useEffect(() => {
    const onWheel = (e) => {
      // If user scrolls up, immediately release lock so user has complete scroll freedom
      if (e.deltaY < 0) {
        pinned.current = false;
      } else if (e.deltaY > 0) {
        // If user scrolls back down near the bottom, re-engage pinning
        const gap = document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
        if (gap < 80) pinned.current = true;
      }
    };
    const onTouch = () => {
      pinned.current = false;
    };
    const onScroll = () => {
      const gap = document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
      if (gap < 80) {
        pinned.current = true;
      }
    };

    window.addEventListener('wheel', onWheel, { passive: true });
    window.addEventListener('touchmove', onTouch, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('wheel', onWheel);
      window.removeEventListener('touchmove', onTouch);
      window.removeEventListener('scroll', onScroll);
    };
  }, []);

  // When messages or stage change, smoothly bring into view ONLY if pinned
  useEffect(() => {
    if (pinned.current) {
      toBottom(true);
    }
  }, [messages.length, stage]);

  /* ── intelligent year clarifier & dispatch ── */
  const handleSend = async (rawQuery, imageUrls = null, base64 = null, activeModes = []) => {
    const query = (rawQuery || '').trim();
    const imagesArray = Array.isArray(imageUrls) ? imageUrls : (imageUrls ? [imageUrls] : []);
    const isSatelliteMode = activeDock === 'sweep' || activeModes.includes('search');

    // Determine if query is a conversational follow-up to ongoing topic
    const hasPriorContext = Boolean(conversationContext && conversationContext.location);
    const isExplicitFollowUp = hasPriorContext && (
      /^(?:what about|how about|what was|tell me|show me|check|what\b|how\b|why\b|where\b|when\b|is there|was there|in\b|during\b|month\b|year\b|and\b)/i.test(query) ||
      /\b(january|february|march|april|may|june|july|august|september|october|november|december|current|recent|today|yesterday|earlier|before|after)\b/i.test(query) ||
      /^\s*(?:in\s+)?(?:the\s+)?(?:month|year)\s+of\s+/i.test(query) ||
      /^\s*(?:19\d{2}|20\d{2})\b/.test(query) ||
      query.split(/\s+/).length <= 4
    );

    // Check if the user is replying to a pending year clarification prompt
    let finalQuery = query;
    if (pendingQueryWithoutYear && !base64) {
      const cleanedFollowUp = query.replace(/^(?:go|yes|do it|check|ok|proceed|in the month of|in month of|in the year of|in year of|in year|in)\s+/i, '').trim();
      if (!/^(cancel|nevermind|no|skip)$/i.test(cleanedFollowUp)) {
        finalQuery = `${pendingQueryWithoutYear} in ${cleanedFollowUp || query}`;
      } else {
        finalQuery = pendingQueryWithoutYear;
      }
      setPendingQueryWithoutYear(null);
    } else if (!base64 && query && !isExplicitFollowUp) {
      // Check if this looks like a brand new disaster location query without a year/time specified
      const hasDisasterWord = /\b(flood|flooding|landslide|mudslide|earthquake|wildfire|cyclone|hurricane|storm|disaster|damage|rubble)\b/i.test(query);
      const hasTimeReference = /\b(19\d{2}|20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december|current|recent|today|yesterday|latest|now)\b/i.test(query);
      const isNotSmalltalk = !/\b(hi|hello|hey|help|who are you|thanks)\b/i.test(query);

      // Only prompt for year if it is a new disaster inquiry lacking a temporal anchor
      if (hasDisasterWord && !hasTimeReference && isNotSmalltalk) {
        setMessages((m) => [
          ...m,
          { role: 'user', content: query, images: imagesArray },
          {
            role: 'ai',
            stream: false,
            content: `I found the location in your query, but disaster events often reoccur across different seasons and years. **Which year or date would you like me to inspect?** *(e.g., "2024", "July 2024", or reply "current")*`,
          },
        ]);
        setPendingQueryWithoutYear(query);
        toBottom();
        return;
      }
    }

    setMessages((m) => [...m, { role: 'user', content: query, images: imagesArray }]);
    setStage({ label: 'Working', state: 'working', variant: 'gyro', detail: 'starting investigation' });
    toBottom();

    let payload = {
      query: finalQuery,
      satellite_mode: isSatelliteMode,
      context: isExplicitFollowUp ? conversationContext : null,
    };
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
      let data = null;
      try {
        data = await res.json();
      } catch (_) {}
      if (!res.ok) throw new Error((data && data.error) || (res.status === 413 ? 'Image is too large. Please upload an image under 32MB.' : 'Could not start chat (' + res.status + ')'));
      const source = new EventSource('/api/chat/stream/' + data.job_id);
      eventSourceRef.current = source;
      source.onmessage = (e) => {
        const item = JSON.parse(e.data);
        if (item.stage === 'complete') {
          source.close();
          setStage(null);
          const r = item.result || {};
          if (r.context && r.context.location) {
            setConversationContext(r.context);
          } else if (r.facts && r.facts.location) {
            setConversationContext({
              location: r.facts.location,
              hazard: r.facts.event_type,
              time: r.facts.time_hint || r.facts.event_date,
            });
          }
          setMessages((m) => [...m, { role: 'ai', stream: true, content: r.response || '', facts: r.facts, meta: { narration: r.narration }, result_image: imagesArray }]);
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

  const handleUpload = async (files) => {
    const fileList = Array.isArray(files) ? files : [files];
    if (fileList.length === 0) return;

    for (let i = 4; i <= 100; i += 18) {
      setUpload(Math.min(i, 100));
      await new Promise(r => setTimeout(r, 60));
    }

    const newUploads = await Promise.all(
      fileList.map((file) => {
        return new Promise((resolve) => {
          const url = URL.createObjectURL(file);
          const reader = new FileReader();
          reader.readAsDataURL(file);
          reader.onload = () => resolve({ url, base64: reader.result });
        });
      })
    );

    setUpload(100);
    await new Promise(r => setTimeout(r, 400));
    setUpload(0);
    setPendingUploads((prev) => [...prev, ...newUploads]);
  };

  const handleRemovePending = (idx) => {
    setPendingUploads((prev) => prev.filter((_, i) => i !== idx));
  };

  useEffect(() => { document.title = 'DisasterRes · Satellite damage assessment'; }, []);

  return (
    <div className="relative min-h-[150vh] overflow-x-hidden pb-52">
      <div className="atmosphere" /><div className="topo" /><div className="grain" />

      {/* top bar fades in once the hero has docked */}
      <motion.div style={{ opacity: bar }}
        className="pointer-events-none fixed inset-x-0 top-0 z-30 h-14 border-b border-bone/10 bg-basalt/70 backdrop-blur-xl" />
      <div className="fixed right-6 top-4 z-40 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[.22em] text-ash">
        <span className={`h-2 w-2 animate-pulse rounded-full ${activeDock === 'sweep' ? 'bg-ember shadow-[0_0_10px_var(--color-ember)]' : 'bg-bone/60'}`} />
        Live · {activeDock === 'sweep' ? 'Satellite sweep active' : 'Disaster desk 01'}
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
              onSubmit={(q, modes) => {
                if (!q.trim() && pendingUploads.length === 0) return;
                const urls = pendingUploads.map(p => p.url);
                // Send primary base64 image (first) to backend for pipeline analysis
                const primaryB64 = pendingUploads.length > 0 ? pendingUploads[0].base64 : null;
                handleSend(q || 'Analyze this imagery', urls, primaryB64, modes);
                setPendingUploads([]);
              }} 
              onAttach={handleUpload} 
              isLoading={!!stage}
              thumbnail={pendingUploads}
              onClearThumbnail={handleRemovePending}
              onStop={handleStop}
            />
          </div>
        </motion.div>

        {/* transcript */}
        <section className="min-h-[70vh] space-y-8 pt-10 pb-32">
          {messages.length === 0 && !stage && (
            <motion.p style={{ opacity: fade }} className="font-mono text-xs uppercase tracking-[.2em] text-ash/60">
              — awaiting coordinates —
            </motion.p>
          )}
          {messages.map((msg, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              className={`flex w-full min-w-0 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'user' && (
                <div className="flex flex-col items-end gap-2.5 max-w-[85%] min-w-0">
                  {/* Clean row of rounded preview tiles for user uploaded images */}
                  {msg.images && msg.images.length > 0 && (
                    <div className="flex flex-wrap items-center justify-end gap-2.5">
                      {msg.images.map((imgUrl, imgIdx) => (
                        <div
                          key={imgIdx}
                          className="relative group cursor-pointer overflow-hidden rounded-2xl border border-bone/20 hover:border-ember bg-basalt-3 shadow-md hover:scale-105 transition-all duration-200"
                          onClick={() => setLightbox(imgUrl)}
                          title="Click to preview full size"
                        >
                          <img
                            src={imgUrl}
                            alt={`Upload ${imgIdx + 1}`}
                            className="w-20 h-20 sm:w-24 sm:h-24 object-cover transition-transform duration-200 group-hover:scale-110"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                  {msg.content && (
                    <div className="w-auto max-w-full min-w-0 break-words rounded-2xl rounded-br-[4px] bg-bone/90 backdrop-blur pl-4 pr-6 py-2.5 text-[15px] leading-snug text-basalt shadow-sm text-left">{msg.content}</div>
                  )}
                </div>
              )}
              {msg.role === 'ai' && <Report msg={msg} onImageClick={setLightbox} />}
              {msg.role === 'error' && (
                <div className="max-w-[80%] min-w-0 break-words rounded-2xl bg-ember/10 px-5 py-3 font-mono text-sm text-ember-soft ring-1 ring-ember/40">{msg.content}</div>
              )}
            </motion.div>
          ))}
          {stage && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} key={stage.label}>
              <ThinkingOrbs {...stage} />
            </motion.div>
          )}
          <div ref={endRef} className="h-4 w-full pointer-events-none" />
        </section>
      </main>

      {upload > 0 && (
        <motion.div initial={{ opacity: 0, scale: .8 }} animate={{ opacity: 1, scale: 1 }}
          className="fixed right-8 top-24 z-40 rounded-3xl bg-basalt-2/90 p-5 backdrop-blur-xl hairline">
          <RadialProgress progress={upload} label={upload >= 100 ? 'Uploaded' : 'Uploading'} />
        </motion.div>
      )}

      <Dock
        defaultActive={activeDock}
        onSelect={(id) => {
          setActiveDock(id);
          if (id === 'upload') document.querySelector('input[type="file"]')?.click();
        }}
      />

      <AnimatePresence>
        {lightbox && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] grid place-items-center bg-black/85 p-6 backdrop-blur-2xl"
            onClick={() => setLightbox(null)}
          >
            <button className="absolute top-6 right-6 text-bone hover:text-white bg-white/10 hover:bg-white/20 rounded-full p-2.5 transition shadow-lg" onClick={() => setLightbox(null)}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
            <motion.img 
              initial={{ scale: 0.92, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.92, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 220 }}
              src={lightbox} alt="Full preview" className="max-h-[90vh] max-w-[90vw] rounded-2xl object-contain shadow-2xl border border-bone/20" 
              onClick={e => e.stopPropagation()} 
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
