package fakeengine

import (
	"net/http"
	"strings"
)

// messages answers the Anthropic Messages API with the same event sequence the native server sends.
func (e *Engine) messages(w http.ResponseWriter, r *http.Request, req map[string]any) {
	toks := replyTokens(req, "max_tokens")
	prompt := promptTokens(req)
	tl := e.tally(prompt, len(toks), streaming(req))
	defer tl.finish()
	id := e.nextID("msg")
	stop := "end_turn"
	if len(toks) < len(words) {
		stop = "max_tokens"
	}
	if !streaming(req) {
		if !e.pause(r.Context()) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"id": id, "type": "message", "role": "assistant", "model": e.opts.Name,
			"content": []map[string]any{{"type": "text", "text": strings.Join(toks, "")}}, "stop_reason": stop, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": prompt, "output_tokens": len(toks), "cache_read_input_tokens": 0}})
		return
	}
	s := newSSE(w, r, e.opts.Delay)
	start := map[string]any{"type": "message_start", "message": map[string]any{"id": id, "type": "message", "role": "assistant",
		"model": e.opts.Name, "content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": prompt, "output_tokens": 0, "cache_read_input_tokens": 0}}}
	if !s.send("message_start", start) {
		return
	}
	if !s.send("content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "text", "text": ""}}) {
		return
	}
	for _, t := range toks {
		tl.token()
		if !s.send("content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": t}}) {
			return
		}
	}
	s.send("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	s.send("message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": len(toks)}})
	s.send("message_stop", map[string]any{"type": "message_stop"})
}
