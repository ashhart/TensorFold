import { Component, type ErrorInfo, type ReactNode } from 'react'

/** A screen that throws while rendering shows what happened and a way back, instead of blanking the whole page. */
export default class ErrorBoundary extends Component<{ name: string; children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error(`${this.props.name} failed to render`, error, info.componentStack) }
  componentDidUpdate(prev: { name: string }) { if (prev.name !== this.props.name && this.state.error) this.setState({ error: null }) }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <section>
        <div className="notice bad" role="alert">
          <b>{this.props.name} could not be shown</b>: {String(this.state.error.message || this.state.error)}.{' '}
          <button className="btn sm" onClick={() => this.setState({ error: null })}>Try again</button>{' '}
          <button className="btn sm quiet" onClick={() => location.reload()}>Reload the page</button>
        </div>
      </section>
    )
  }
}
