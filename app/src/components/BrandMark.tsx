import mark from '../assets/tensorfold-mark.png?inline'

// The TensorFold fold. `tile` sets it on the brand navy (app-icon style) for avatars and
// headers; bare, it floats on the current surface (the home splash).
export function BrandMark({ size = 36, tile = false, className = '' }: { size?: number; tile?: boolean; className?: string }) {
  const img = <img src={mark} alt="" draggable={false} style={{ width: tile ? size * 0.82 : size }} className="pointer-events-none select-none" />
  if (!tile) return <span className={`inline-flex items-center justify-center ${className}`}>{img}</span>
  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center overflow-hidden rounded-[28%] bg-[var(--brand-navy)] ${className}`}
      style={{ width: size, height: size }}
    >
      {img}
    </span>
  )
}
