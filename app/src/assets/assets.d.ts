// Vite's `?inline` import: the asset as a data: URI (the renderer CSP allows data: images).
declare module '*.png?inline' {
  const src: string
  export default src
}
