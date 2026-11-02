package fakeengine

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// respond implements POST /v1/responses with the native server's store rules for scoped keys.
func (e *Engine) respond(w http.ResponseWriter, r *http.Request, req map[string]any) {
	store := true
	if v, ok := req["store"].(bool); ok {
		store = v
	}
	prev, _ := req["previous_response_id"].(string)
	if k, ok := keyFrom(r.Context()); ok && !k.full() {
		if store {
			writeErr(w, http.StatusForbidden, "insufficient_scope", "a scoped key must send store: false")
			return
		}
		if prev != "" {
			writeErr(w, http.StatusForbidden, "insufficient_scope", "a scoped key cannot send previous_response_id")
			return
		}
	}
	if prev != "" {
		e.mu.Lock()
		_, ok := e.responses[prev]
		e.mu.Unlock()
		if !ok {
			writeErr(w, http.StatusBadRequest, "previous_response_not_found", "previous_response_id is not stored")
			return
		}
	}
	toks := replyTokens(req, "max_output_tokens")
	prompt := promptTokens(req)
	tl := e.tally(prompt, len(toks), streaming(req))
	defer tl.finish()
	id := e.nextID("resp")
	text := strings.Join(toks, "")
	status := "completed"
	if len(toks) < len(words) {
		status = "incomplete"
	}
	resp := map[string]any{"id": id, "object": "response", "created_at": time.Now().Unix(), "model": e.opts.Name, "status": status,
		"output": []map[string]any{{"type": "message", "id": "msg_" + id, "role": "assistant", "status": "completed",
			"content": []map[string]any{{"type": "output_text", "text": text, "annotations": []any{}}}}},
		"usage": map[string]any{"input_tokens": prompt, "output_tokens": len(toks), "total_tokens": prompt + len(toks),
			"input_tokens_details": map[string]any{"cached_tokens": 0}, "output_tokens_details": map[string]any{"reasoning_tokens": 0}}}
	if store {
		b, _ := json.Marshal(resp)
		e.mu.Lock()
		e.responses[id] = b
		e.mu.Unlock()
	}
	if !streaming(req) {
		if !e.pause(r.Context()) {
			return
		}
		writeJSON(w, http.StatusOK, resp)
		return
	}
	s := newSSE(w, r, e.opts.Delay)
	seq := 0
	ev := func(kind string, fields map[string]any) bool {
		fields["type"], fields["sequence_number"] = kind, seq
		seq++
		return s.send(kind, fields)
	}
	shell := map[string]any{"id": id, "object": "response", "model": e.opts.Name, "status": "in_progress", "output": []any{}}
	if !ev("response.created", map[string]any{"response": shell}) || !ev("response.in_progress", map[string]any{"response": shell}) {
		return
	}
	item := map[string]any{"type": "message", "id": "msg_" + id, "role": "assistant", "status": "in_progress", "content": []any{}}
	ev("response.output_item.added", map[string]any{"output_index": 0, "item": item})
	ev("response.content_part.added", map[string]any{"output_index": 0, "content_index": 0, "item_id": "msg_" + id,
		"part": map[string]any{"type": "output_text", "text": ""}})
	for _, t := range toks {
		tl.token()
		if !ev("response.output_text.delta", map[string]any{"output_index": 0, "content_index": 0, "item_id": "msg_" + id, "delta": t}) {
			return
		}
	}
	ev("response.output_text.done", map[string]any{"output_index": 0, "content_index": 0, "item_id": "msg_" + id, "text": text})
	ev("response.content_part.done", map[string]any{"output_index": 0, "content_index": 0, "item_id": "msg_" + id,
		"part": map[string]any{"type": "output_text", "text": text}})
	ev("response.output_item.done", map[string]any{"output_index": 0, "item": resp["output"].([]map[string]any)[0]})
	final := "response.completed"
	if status == "incomplete" {
		final = "response.incomplete"
	}
	ev(final, map[string]any{"response": resp})
}

// storedResponse serves GET and DELETE /v1/responses/{id}; scoped keys never reach stored responses.
func (e *Engine) storedResponse(w http.ResponseWriter, r *http.Request) {
	if k, ok := keyFrom(r.Context()); ok && !k.full() {
		writeErr(w, http.StatusForbidden, "insufficient_scope", "stored responses need a full key")
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/v1/responses/")
	e.mu.Lock()
	body, ok := e.responses[id]
	if ok && r.Method == http.MethodDelete {
		delete(e.responses, id)
	}
	e.mu.Unlock()
	if !ok {
		writeErr(w, http.StatusNotFound, "not_found", "no stored response with that id")
		return
	}
	if r.Method == http.MethodDelete {
		writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "response.deleted", "deleted": true})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(body)
}
