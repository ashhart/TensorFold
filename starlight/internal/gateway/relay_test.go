package gateway

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/providers"
)

func sse(events ...string) *http.Response {
	body := strings.Join(events, "\n\n") + "\n\n"
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(body))}
}

func TestRelayOpenAIStreamHidesInjectedUsage(t *testing.T) {
	resp := sse(`data: {"choices":[{"delta":{"content":"hi"}}]}`, `data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":3}}}`, `data: [DONE]`)
	w := httptest.NewRecorder()
	out := relay(context.Background(), w, resp, providers.ChatCompletions, true, httpx.OpenAI, time.Now())
	if out.outcome != "ok" || out.usage.Prompt != 7 || out.usage.Completion != 2 || out.usage.Cached != 3 || !out.usage.Observed {
		t.Fatalf("out = %+v", out)
	}
	if strings.Contains(w.Body.String(), "usage") || !strings.Contains(w.Body.String(), "[DONE]") || !strings.Contains(w.Body.String(), `"hi"`) {
		t.Fatalf("client saw %q", w.Body.String())
	}
	resp = sse(`data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`, `data: [DONE]`)
	w = httptest.NewRecorder()
	relay(context.Background(), w, resp, providers.ChatCompletions, false, httpx.OpenAI, time.Now())
	if !strings.Contains(w.Body.String(), "usage") {
		t.Fatal("requested usage chunk was hidden")
	}
}

func TestRelayAnthropicAndResponsesStreams(t *testing.T) {
	resp := sse("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":90,\"output_tokens\":1}}}",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"ok\"}}",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}", "event: message_stop\ndata: {\"type\":\"message_stop\"}")
	out := relay(context.Background(), httptest.NewRecorder(), resp, providers.Messages, false, httpx.Anthropic, time.Now())
	if out.usage.Prompt != 100 || out.usage.Cached != 90 || out.usage.Completion != 42 || !out.usage.Observed {
		t.Fatalf("anthropic usage %+v", out.usage)
	}
	resp = sse(`data: {"type":"response.created","response":{"id":"resp_abc"}}`, `data: {"type":"response.output_text.delta","delta":"x"}`,
		`data: {"type":"response.completed","response":{"id":"resp_abc","usage":{"input_tokens":5,"output_tokens":6,"input_tokens_details":{"cached_tokens":1}}}}`)
	out = relay(context.Background(), httptest.NewRecorder(), resp, providers.Responses, false, httpx.OpenAI, time.Now())
	if out.responseID != "resp_abc" || out.usage.Prompt != 5 || out.usage.Completion != 6 || out.usage.Cached != 1 {
		t.Fatalf("responses %+v", out)
	}
}

type failing struct{ r io.Reader }

func (f failing) Read(p []byte) (int, error) {
	n, err := f.r.Read(p)
	if err == io.EOF {
		return n, errors.New("connection reset")
	}
	return n, err
}

func TestRelayInterruptedStreamIsUnobserved(t *testing.T) {
	resp := &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}},
		Body: io.NopCloser(failing{strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n")})}
	w := httptest.NewRecorder()
	out := relay(context.Background(), w, resp, providers.ChatCompletions, true, httpx.OpenAI, time.Now())
	if out.outcome != "interrupted" || out.usage.Observed || !strings.Contains(w.Body.String(), "stream_interrupted") {
		t.Fatalf("out = %+v body %q", out, w.Body.String())
	}
}

func TestRelayJSONErrorsPassThrough(t *testing.T) {
	resp := &http.Response{StatusCode: 400, Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{"error":{"message":"context_length_exceeded"}}`))}
	w := httptest.NewRecorder()
	out := relay(context.Background(), w, resp, providers.ChatCompletions, false, httpx.OpenAI, time.Now())
	if w.Code != 400 || out.outcome != "client_error" || !out.usage.Observed || out.usage.Prompt != 0 {
		t.Fatalf("out = %+v code %d", out, w.Code)
	}
}
