import { Orb } from './orb';

/* "Chat line" thinking-orbs status: orb + shimmering verb, like an agent transcript. */
export function ThinkingOrbs({ label = 'Working', detail, state = 'working', variant }) {
  return (
    <div className="flex items-center gap-3 text-[15px] leading-[1.65]">
      <span className="flex h-[1.65em] w-5 shrink-0 items-center text-ember">
        <Orb state={state} variant={variant} size={20} label={label} />
      </span>
      <span className="flex flex-wrap items-center gap-x-2">
        <span className="shimmer font-medium">{label}</span>
        {detail && <span className="font-mono text-xs text-ash truncate">{detail}</span>}
      </span>
    </div>
  );
}
