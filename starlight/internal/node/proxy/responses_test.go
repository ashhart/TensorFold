package proxy

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
	"github.com/ashhart/TensorFold/starlight/internal/node/nodetest"
)

func TestKeyForChoosesLeastPrivilege(t *testing.T) {
	keys := engine.Keys{Inference: "inf", Responses: "full"}
	cases := []struct {
		method, path, body, want string
	}{
		{"POST", "/v1/responses", `{"input":"hi"}`, "full"},
		{"POST", "/v1/responses", `{"input":"hi","store":true}`, "full"},
		{"POST", "/v1/responses", `{"input":"hi","store":false}`, "inf"},
		{"POST", "/v1/responses", `{"input":"hi","store":false,"previous_response_id":"resp_1"}`, "full"},
		{"POST", "/v1/responses", `{"input":"hi","store":false,"previous_response_id":""}`, "inf"},
		{"POST", "/v1/responses", `not json`, "inf"},
		{"GET", "/v1/responses/resp_1", ``, "full"},
		{"DELETE", "/v1/responses/resp_1", ``, "full"},
		{"POST", "/v1/chat/completions", `{"store":true}`, "inf"},
		{"POST", "/v1/messages", `{}`, "inf"},
	}
	for _, c := range cases {
		if got := keyFor(c.method, c.path, []byte(c.body), keys); got != c.want {
			t.Errorf("%s %s %s = %s, want %s", c.method, c.path, c.body, got, c.want)
		}
	}
}

func (f *fixture) send(t *testing.T, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	c := f.claims(path)
	c.Method = method
	w := httptest.NewRecorder()
	f.srv.ServeHTTP(w, f.request(t, method, path, body, nodetest.Permit(t, f.signer, c, []byte(body))))
	return w
}

func (f *fixture) label() string { return f.eng.Stats().Last.Label }

func TestStoredResponsesUseTheFullKeyOnly(t *testing.T) {
	f := newFixture(t, 0)
	f.tgt.slots = 4
	w := f.send(t, "POST", "/v1/responses", `{"input":"hello"}`)
	if w.Code != 200 || f.label() != "starlight-responses" {
		t.Fatalf("store by default: %d via %s: %s", w.Code, f.label(), w.Body)
	}
	var stored struct{ ID string }
	_ = json.Unmarshal(w.Body.Bytes(), &stored)
	if w := f.send(t, "POST", "/v1/responses", `{"input":"hello","store":false}`); w.Code != 200 || f.label() != "starlight" {
		t.Fatalf("store false: %d via %s", w.Code, f.label())
	}
	next := `{"input":"and then","store":false,"previous_response_id":"` + stored.ID + `"}`
	if w := f.send(t, "POST", "/v1/responses", next); w.Code != 200 || f.label() != "starlight-responses" {
		t.Fatalf("continuation: %d via %s: %s", w.Code, f.label(), w.Body)
	}
	if w := f.send(t, "GET", "/v1/responses/"+stored.ID, ""); w.Code != 200 || f.label() != "starlight-responses" {
		t.Fatalf("get: %d via %s", w.Code, f.label())
	}
	if w := f.send(t, "DELETE", "/v1/responses/"+stored.ID, ""); w.Code != 200 || f.label() != "starlight-responses" {
		t.Fatalf("delete: %d via %s", w.Code, f.label())
	}
	if w := f.send(t, "GET", "/v1/responses/"+stored.ID, ""); w.Code != 404 {
		t.Fatalf("get after delete: %d", w.Code)
	}
	if w := f.send(t, "POST", "/v1/chat/completions", chatBody); w.Code != 200 || f.label() != "starlight" {
		t.Fatalf("chat must keep the inference key: %d via %s", w.Code, f.label())
	}
}

func TestInferenceKeyAloneCannotStore(t *testing.T) {
	f := newFixture(t, 0)
	f.tgt.keys.Responses = f.tgt.keys.Inference
	if w := f.send(t, "POST", "/v1/responses", `{"input":"hello"}`); w.Code != 403 {
		t.Fatalf("the engine's scope rules should refuse a stored response with the inference key: %d", w.Code)
	}
}
