/** Fingertip over a light source and sensor: what a PPG sensor does, in one picture. */
export function PpgDiagram({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 220 130" className={className} role="img" aria-label="A fingertip resting on a green light and a light sensor">
      <path d="M20 70 Q20 22 80 22 L200 22 L200 70 Z" className="fill-subtle stroke-rule" strokeWidth="1.5" />
      <path d="M40 46 Q60 38 80 46 T120 46 T160 46 T200 46" fill="none" className="stroke-ref-grey" strokeWidth="1.5" strokeDasharray="3 4" />
      <rect x="20" y="70" width="180" height="16" rx="3" className="fill-surface stroke-rule" strokeWidth="1.5" />
      <rect x="70" y="72" width="18" height="10" rx="2" className="fill-led" />
      <rect x="120" y="72" width="18" height="10" rx="2" className="fill-ink" />
      <path d="M79 70 L92 42 L125 42 L129 70" fill="none" className="stroke-led" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <text x="79" y="104" textAnchor="middle" className="fill-muted-foreground text-[11px]">light</text>
      <text x="129" y="104" textAnchor="middle" className="fill-muted-foreground text-[11px]">sensor</text>
    </svg>
  );
}
