package fakeengine

import (
	"net/http"
	"strings"
	"time"
)

func (e *Engine) usage(prompt, completion int) map[string]any {
	return map[string]any{"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
		"prompt_tokens_details": map[string]any{"cached_tokens": 0}, "completion_tokens_details": map[string]any{"reasoning_tokens": 0}}
}

// chat answers like the native server: streamed chunks with usage on the finish chunk, or a separate usage chunk.
func (e *Engine) chat(w http.ResponseWriter, r *http.Request, req map[string]any) {
	toks := replyTokens(req, "max_tokens", "max_completion_tokens")
	prompt := promptTokens(req)
	tl := e.tally(prompt, len(toks), streaming(req))
	defer tl.finish()
	id, created := e.nextID("chatcmpl"), time.Now().Unix()
	finish := "stop"
	if len(toks) < len(words) {
		finish = "length"
	}
	if !streaming(req) {
		if !e.pause(r.Context()) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "chat.completion", "created": created, "model": e.opts.Name,
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": strings.Join(toks, "")},
				"finish_reason": finish}}, "usage": e.usage(prompt, len(toks))})
		return
	}
	includeUsage := false
	if so, ok := req["stream_options"].(map[string]any); ok {
		includeUsage, _ = so["include_usage"].(bool)
	}
	s := newSSE(w, r, e.opts.Delay)
	chunk := func(delta map[string]any, fin any) map[string]any {
		return map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": e.opts.Name,
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": fin}}}
	}
	if !s.send("", chunk(map[string]any{"role": "assistant", "content": ""}, nil)) {
		return
	}
	for _, t := range toks {
		tl.token()
		if !s.send("", chunk(map[string]any{"content": t}, nil)) {
			return
		}
	}
	last := chunk(map[string]any{}, finish)
	if includeUsage {
		s.send("", last)
		s.send("", map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": e.opts.Name,
			"choices": []any{}, "usage": e.usage(prompt, len(toks))})
	} else {
		last["usage"] = e.usage(prompt, len(toks))
		s.send("", last)
	}
	s.send("", "[DONE]")
}

func (e *Engine) completion(w http.ResponseWriter, r *http.Request, req map[string]any) {
	toks := replyTokens(req, "max_tokens")
	prompt := promptTokens(req)
	tl := e.tally(prompt, len(toks), streaming(req))
	defer tl.finish()
	id, created := e.nextID("cmpl"), time.Now().Unix()
	if !streaming(req) {
		if !e.pause(r.Context()) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "text_completion", "created": created, "model": e.opts.Name,
			"choices": []map[string]any{{"index": 0, "text": strings.Join(toks, ""), "finish_reason": "stop"}}, "usage": e.usage(prompt, len(toks))})
		return
	}
	s := newSSE(w, r, e.opts.Delay)
	for _, t := range toks {
		tl.token()
		if !s.send("", map[string]any{"id": id, "object": "text_completion", "created": created, "model": e.opts.Name,
			"choices": []map[string]any{{"index": 0, "text": t, "finish_reason": nil}}}) {
			return
		}
	}
	s.send("", map[string]any{"id": id, "object": "text_completion", "created": created, "model": e.opts.Name,
		"choices": []map[string]any{{"index": 0, "text": "", "finish_reason": "stop"}}, "usage": e.usage(prompt, len(toks))})
	s.send("", "[DONE]")
}

func (e *Engine) embeddings(w http.ResponseWriter, req map[string]any) {
	var inputs []string
	switch v := req["input"].(type) {
	case string:
		inputs = []string{v}
	case []any:
		for _, x := range v {
			if s, ok := x.(string); ok {
				inputs = append(inputs, s)
			}
		}
	}
	if len(inputs) == 0 {
		writeErr(w, http.StatusBadRequest, "invalid_input", "input must be a string or a list of strings")
		return
	}
	data := make([]map[string]any, len(inputs))
	total := 0
	for i, in := range inputs {
		vec := make([]float64, 8)
		for j := range vec {
			vec[j] = float64((len(in)+j*7)%13) / 13
		}
		total += len(strings.Fields(in))
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": vec}
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data, "model": e.opts.Name,
		"usage": map[string]any{"prompt_tokens": total, "total_tokens": total}})
}
