package fleetpages_test

import (
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"testing"
)

var (
	assetLink  = regexp.MustCompile(`/static/fleet\.css\?v=\w+`)
	liveScript = regexp.MustCompile(`<script src="/static/console-live\.js\?v=\w+" defer></script>`)
)

func TestEveryPageRendersForAnOwner(t *testing.T) {
	r := newRig(t)
	owner := r.user("owner@example.com", "owner", "")
	pages := map[string][]string{
		"/console":                   {"Generation now", "87.5", "studio-1", "Who's connected", "Take offline", "Approve"},
		"/console/projects":          {"Research", "New project"},
		"/console/projects/research": {"Members", "Service accounts and keys", "Usage, last 30 days"},
		"/console/models":            {"team/coder", `data-model-id="` + r.model + `"`, "Add a model"},
		"/console/models/" + r.model: {"coder-mac", "New deployment", "Frontier routes", "Global aliases", "Save model"},
		"/console/models/" + r.model + "/deployments/new": {"New deployment", "Model weights", "Create deployment"},
		"/console/deployments/coder-mac":                  {"studio-1", "Save deployment", "Retire deployment", "path:/srv/coder"},
		"/console/nodes":                                  {"studio-1", "Apple M3 Ultra", "Enrol a node", "Drain", "Revoke…"},
		"/console/nodes/studio-1":                         {"Placement", "Revoke node", "coder-mac"},
		"/console/providers":                              {"Add a provider"},
		"/console/limits":                                 {"Set a limit"},
		"/console/usage":                                  {"Download CSV", "Group by"},
		"/console/plans":                                  {"Waiting for approval", "My plans"},
		"/console/subscriptions":                          {"Reclamation report", "Record a subscription"},
	}
	for path, want := range pages {
		code, body := owner.get(path)
		if code != http.StatusOK {
			t.Fatalf("%s answered %d: %s", path, code, body)
		}
		for _, w := range want {
			if !strings.Contains(body, w) {
				t.Errorf("%s is missing %q", path, w)
			}
		}
		if !assetLink.MatchString(body) {
			t.Errorf("%s does not load the versioned fleet stylesheet", path)
		}
		if strings.Contains(body, "data-live-url") && !liveScript.MatchString(body) {
			t.Errorf("%s polls the live feed without loading the live script", path)
		}
	}
}

func TestReadOnlyRolesSeeNoControls(t *testing.T) {
	r := newRig(t)
	auditor := r.user("audit@example.com", "auditor", "")
	checks := map[string][]string{
		"/console":                       {"Take offline", "Drain"},
		"/console/nodes":                 {"Enrol a node", "Take offline", "Revoke…"},
		"/console/nodes/studio-1":        {"Revoke node", "Save placement"},
		"/console/models":                {"Add a model"},
		"/console/models/" + r.model:     {"New deployment", "Save model", "Save route", "Add alias"},
		"/console/deployments/coder-mac": {"Save deployment", "Retire deployment", "Pause"},
		"/console/providers":             {"Add a provider"},
		"/console/subscriptions":         {"Record a subscription", "Free seat"},
	}
	for path, hidden := range checks {
		code, body := auditor.get(path)
		if code != http.StatusOK {
			t.Fatalf("%s answered %d", path, code)
		}
		for _, h := range hidden {
			if strings.Contains(body, h) {
				t.Errorf("%s shows %q to a read-only role", path, h)
			}
		}
	}
	posts := map[string]url.Values{
		"/console/nodes/studio-1/action":        {"action": {"drain"}},
		"/console/nodes/tokens":                 {"pool": {"default"}},
		"/console/models":                       {"name": {"team/new"}},
		"/console/deployments/coder-mac":        {"model_kind": {"path"}, "model_path": {"/srv/x"}},
		"/console/providers":                    {"name": {"openai-main"}, "kind": {"openai"}},
		"/console/limits":                       {"kind": {"concurrency"}, "value": {"1"}},
		"/console/subscriptions":                {"vendor": {"Example"}, "product": {"Seats"}, "seats": {"1"}},
		"/console/deployments/coder-mac/status": {"status": {"paused"}},
	}
	for path, form := range posts {
		if code, _ := auditor.post(path, form); code != http.StatusForbidden {
			t.Errorf("POST %s by an auditor answered %d", path, code)
		}
	}
	if code, _ := auditor.get("/console/limits"); code != http.StatusForbidden {
		t.Errorf("limits page answered %d without limits.write", code)
	}
	if r.audited("node.drain") != 0 || r.scalar("SELECT status FROM deployments WHERE name = 'coder-mac'") != "active" {
		t.Fatal("a forbidden action changed state")
	}
}

func TestMembersGetThePersonalView(t *testing.T) {
	r := newRig(t)
	r.exec("INSERT INTO model_grants (id, model_id) VALUES (gen_random_uuid(), $1)", r.model)
	member := r.user("pat@example.com", "", "member")
	code, body := member.get("/console")
	if code != http.StatusOK || !strings.Contains(body, "Your requests today") || strings.Contains(body, "studio-1") || !strings.Contains(body, "team/coder") {
		t.Fatalf("personal overview %d: %s", code, body)
	}
	for _, path := range []string{"/console/nodes", "/console/models", "/console/providers", "/console/subscriptions", "/console/limits",
		"/console/deployments/coder-mac", "/console/nodes/studio-1"} {
		if code, _ := member.get(path); code != http.StatusForbidden {
			t.Errorf("%s answered %d to a project member", path, code)
		}
	}
	code, body = member.get("/console/live.json")
	var snap map[string]json.RawMessage
	if code != http.StatusOK || json.Unmarshal([]byte(body), &snap) != nil {
		t.Fatalf("live.json %d: %s", code, body)
	}
	if string(snap["nodes"]) != "[]" || string(snap["models"]) != "[]" || string(snap["in_flight"]) != "[]" || snap["personal"] == nil ||
		strings.Contains(body, "studio-1") || strings.Contains(body, "tokens_per_second") {
		t.Fatalf("live.json leaked fleet data to a member: %s", body)
	}
	if code, body := member.get("/console/projects/research"); code != http.StatusOK || strings.Contains(body, "Save settings") {
		t.Fatalf("member project page %d", code)
	}
	if code, body := member.get("/console/usage"); code != http.StatusOK || strings.Contains(body, "Download CSV") {
		t.Fatalf("member usage page %d", code)
	}
	if code, _ := member.get("/console/usage.csv"); code != http.StatusForbidden {
		t.Fatalf("member CSV export answered %d", code)
	}
}
