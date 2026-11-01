import { brandOf, customBrand, hardwareOf, type Brand, type BrandKind } from './brand'

// the makers' logos (assets/models/README.md says where each one comes from)
const byName = (files: Record<string, string>): Record<string, string> =>
  Object.fromEntries(Object.entries(files).map(([path, url]) => [path.split('/').pop()!.replace(/\.(svg|png)$/, ''), url]))
const LOGOS = byName(import.meta.glob('./assets/models/*.{svg,png}', { eager: true, query: '?url', import: 'default' }))
// the hardware makers' logos (assets/hardware/README.md)
const HARDWARE = byName(import.meta.glob('./assets/hardware/*.{svg,png}', { eager: true, query: '?url', import: 'default' }))

type Size = 'xs' | 'sm' | 'lg'

/** A maker's logo: the one given in Settings, else TensorFold Web's own; undefined when there is neither. */
export function logoOf(id: string, kind: BrandKind): string | undefined {
  return customBrand(id, kind)?.logo ?? (kind === 'model' ? LOGOS : HARDWARE)[id]
}

/** A maker's logo on a light tile, or its initial when TensorFold Web holds no logo for it. */
export function Tile({ b, url, size }: { b: Brand; url: string | undefined; size: Size }) {
  return url ? <span className={`toollogo ${size}`} title={b.label}><img src={url} alt={b.label} /></span>
    : <span className={`toollogo ${size} initial`} title={b.label} role="img" aria-label={b.label}>{b.label[0]}</span>
}

/** A model's logo, from any of the names it goes by; nothing when its maker is not known. */
export function ModelLogo({ names, size = 'sm' }: { names: (string | null | undefined)[]; size?: Size }) {
  const b = brandOf(...names)
  return b ? <Tile b={b} url={logoOf(b.id, 'model')} size={size} /> : null
}

/** A machine's maker logo (Apple, NVIDIA, AMD) from its chip or backend; nothing when the maker is not known. */
export function HardwareLogo({ names, size = 'sm' }: { names: (string | null | undefined)[]; size?: Size }) {
  const b = hardwareOf(...names)
  return b ? <Tile b={b} url={logoOf(b.id, 'hardware')} size={size} /> : null
}
