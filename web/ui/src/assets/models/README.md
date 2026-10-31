# Model makers' logos

Shown beside a model's name (Home, Models, Chat, Fleet, the sidebar), only to say whose model it is. Every logo
belongs to its owner. `src/brand.ts` says which names map to which logo: a model family's own logo when it has one
(Qwen), else its maker's (NVIDIA for Nemotron, Google for Gemma).

| File | Source |
|---|---|
| nvidia.svg, qwen.svg, deepseek.svg | [Simple Icons](https://simpleicons.org) 16.35.0 (CC0), in the brand colour it gives |
| meta.svg, mistralai.svg, kimi.svg, minimax.svg, huggingface.svg, baidu.svg, bytedance.svg, xiaomi.svg | [Simple Icons](https://simpleicons.org) 16.35.0 (CC0), in the brand colour it gives; taken on 13 October 2026 |
| zai.svg | z.ai, its icon (z-cdn.chatglm.cn/z-ai/static/logo.svg) |
| prism.png | prismml.com, its 32 px favicon |
| google.png | Google's 2025 "G" in its colours (the Google_Favicon_2025 file TensorFold Web's owner supplied), scaled to 96 px |

Taken on 12 October 2026. Google publishes no Gemma icon that a page declares, so Gemma shows Google's.
A maker that `brand.ts` knows but that has no file here (Microsoft, OpenAI, IBM, Ai2, Cohere, Nous Research: Simple
Icons carries none) shows its initial on the same tile.
