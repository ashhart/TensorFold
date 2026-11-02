package e2e_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

func TestStoredResponsesBelongToTheirOwner(t *testing.T) {
	f := startFleet(t)
	resp, body := f.call("POST", "/v1/responses", f.key, `{"model":"qwen3.8-27b","input":"remember the number 7"}`)
	var created struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal([]byte(body), &created)
	if resp.StatusCode != 200 || !strings.HasPrefix(created.ID, "resp") {
		t.Fatalf("store %d %s", resp.StatusCode, body)
	}
	if resp, body := f.call("GET", "/v1/responses/"+created.ID, f.key, ""); resp.StatusCode != 200 || !strings.Contains(body, created.ID) {
		t.Fatalf("owner read %d %s", resp.StatusCode, body)
	}
	resp, body = f.call("POST", "/v1/responses", f.key, `{"model":"qwen3.8-27b","input":"what was it?","previous_response_id":"`+created.ID+`"}`)
	if resp.StatusCode != 200 {
		t.Fatalf("continuation %d %s", resp.StatusCode, body)
	}
	other := f.secondUserKey()
	if resp, _ := f.call("GET", "/v1/responses/"+created.ID, other, ""); resp.StatusCode != 404 {
		t.Fatalf("another user read the response: %d", resp.StatusCode)
	}
	if resp, _ := f.call("POST", "/v1/responses", other, `{"model":"qwen3.8-27b","input":"x","previous_response_id":"`+created.ID+`"}`); resp.StatusCode != 404 {
		t.Fatalf("another user continued the response: %d", resp.StatusCode)
	}
	if resp, body := f.call("DELETE", "/v1/responses/"+created.ID, f.key, ""); resp.StatusCode != 200 {
		t.Fatalf("delete %d %s", resp.StatusCode, body)
	}
	if resp, _ := f.call("GET", "/v1/responses/"+created.ID, f.key, ""); resp.StatusCode != 404 {
		t.Fatalf("deleted response still readable: %d", resp.StatusCode)
	}
	if resp, body := f.call("POST", "/v1/responses", f.key, `{"model":"qwen3.8-27b","input":"stateless","store":false}`); resp.StatusCode != 200 {
		t.Fatalf("store:false %d %s", resp.StatusCode, body)
	}
}

// secondUserKey makes another member of the same project and returns their API key.
func (f *fleet) secondUserKey() string {
	u, err := directory.CreateUser(f.ctx, f.pool, directory.NewUser{Email: "kim@example.com", Source: "jit"})
	if err != nil {
		f.t.Fatal(err)
	}
	exec(f.t, f.pool, "INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'member', 'user', $2, $3)", ids.New(), u.ID, f.project)
	tok, hash := secret.Generate(secret.APIKey)
	exec(f.t, f.pool, `INSERT INTO api_keys (id, public_id, secret_hash, name, owner_type, owner_id, project_id) VALUES ($1,$2,$3,'k','user',$4,$5)`,
		ids.New(), secret.PublicID(tok), hash, u.ID, f.project)
	return tok
}
