# Visualization Widgets

TensorFold can render structured visual artifacts directly in Chat and Code timelines. The first
renderer is dependency-free and supports four widget kinds:

- `flow` - node and edge diagrams for workflows, architecture, and reasoning maps.
- `bar` - quantitative comparisons.
- `stats` - compact metric cards.
- `timeline` - ordered milestones, incidents, or plan history.

## Model Paths

OpenAI-compatible providers with tool calling can use the `visualize` tool. Chat receives only this
tool; Code receives it alongside filesystem and shell tools.

CLI providers that do not expose TensorFold's tool schema can emit a fenced widget block in normal
assistant text:

````markdown
```vontra-widget
{
  "kind": "flow",
  "title": "One Model, Three Machines",
  "nodes": [
    { "id": "gb10", "title": "GB10", "subtitle": "trains base", "tone": "accent", "column": 0, "row": 0 },
    { "id": "base", "title": "base.pt", "subtitle": "the model", "tone": "accent", "column": 1, "row": 0 }
  ],
  "edges": [{ "from": "gb10", "to": "base" }],
  "legend": [{ "label": "the model being built", "tone": "accent" }]
}
```
````

The renderer removes valid widget fences from markdown and replaces them with the same card used by
tool-generated visualizations.

## Shared Contract

Common fields:

```ts
{
  kind: 'flow' | 'bar' | 'stats' | 'timeline'
  title: string
  subtitle?: string
  caption?: string
  legend?: { label: string; tone?: Tone }[]
}
```

Tone values are `accent`, `muted`, `positive`, `warning`, and `danger`.

Flow widgets use:

```ts
{
  nodes: { id: string; title: string; subtitle?: string; tone?: Tone; column?: number; row?: number }[]
  edges?: { from: string; to: string; label?: string }[]
}
```

Bar widgets use:

```ts
{
  bars: { label: string; value: number; max?: number; detail?: string; tone?: Tone }[]
}
```

Stats widgets use:

```ts
{
  metrics: { label: string; value: string; detail?: string; tone?: Tone }[]
}
```

Timeline widgets use:

```ts
{
  events: { title: string; time?: string; detail?: string; tone?: Tone }[]
}
```

## Next Renderer Slices

- Mermaid import/export for sequence diagrams and state machines.
- Vega/Recharts-backed quantitative charts behind the same `bar`/future `line` contracts.
- Export widget to PNG/SVG.
- Add artifact persistence so a widget can be opened, renamed, and reused outside the transcript.
