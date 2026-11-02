package fleetpages_test

import (
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

func TestCatalogueActionsAreAudited(t *testing.T) {
	r := newRig(t)
	owner := r.user("owner@example.com", "owner", "")
	if code, body := owner.post("/console/projects", url.Values{"slug": {"platform"}, "name": {"Platform"}}); code != http.StatusSeeOther {
		t.Fatalf("create project %d: %s", code, body)
	}
	code, page := owner.follow("/console/models", url.Values{"name": {"team/embed"}, "kind": {"embedding"}, "route_policy": {"local_only"},
		"price_input": {"0.02"}})
	if code != http.StatusSeeOther || !strings.Contains(page, "Model added to the catalogue.") || !strings.Contains(page, "local_only") {
		t.Fatalf("create model %d", code)
	}
	if r.scalar("SELECT chargeback->>'input_micros' FROM models WHERE name = 'team/embed'") != "20000" {
		t.Fatal("chargeback price was not stored in micro-dollars")
	}
	for path, form := range map[string]url.Values{
		"/aliases": {"alias": {"coder"}},
		"/grant":   {"everyone": {"on"}, "max_context": {"32768"}},
	} {
		if code, body := owner.post("/console/models/"+r.model+path, form); code != http.StatusSeeOther {
			t.Fatalf("%s %d: %s", path, code, body)
		}
	}
	if code, _ := owner.post("/console/models/not-an-id/aliases", url.Values{"alias": {"x"}}); code != http.StatusNotFound {
		t.Fatalf("unknown model answered %d", code)
	}
	for _, action := range []string{"project.created", "model.created", "model.alias_set", "model.granted"} {
		if r.audited(action) != 1 {
			t.Errorf("%s was not audited once", action)
		}
	}
	if r.scalar("SELECT actor_type || ':' || actor_id FROM audit_events WHERE action = 'model.created'") != "user:"+owner.user.ID {
		t.Fatal("audit events are not attributed to the signed-in user")
	}
}

func TestDeploymentFormValidatesTheSpec(t *testing.T) {
	r := newRig(t)
	owner := r.user("owner@example.com", "owner", "")
	form := url.Values{"name": {"coder-gpu"}, "model_kind": {"hf"}, "model_repo": {"example-org/coder"}, "model_revision": {"main"},
		"backend": {"cuda"}, "pool": {"gpu"}, "mode": {"on_demand"}, "min_replicas": {"0"}, "max_replicas": {"3"}, "selector": {"gpu=b300"}}
	code, body := owner.post("/console/models/"+r.model+"/deployments", form)
	if code != http.StatusUnprocessableEntity || !strings.Contains(body, "40-character commit revision") || !strings.Contains(body, `value="example-org/coder"`) ||
		!strings.Contains(body, "gpu=b300") {
		t.Fatalf("unpinned revision %d: %s", code, body)
	}
	form.Set("model_revision", strings.Repeat("ab", 20))
	form.Set("backend", "tpu")
	if code, body := owner.post("/console/models/"+r.model+"/deployments", form); code != http.StatusUnprocessableEntity || !strings.Contains(body, "unknown backend") {
		t.Fatalf("unknown backend %d", code)
	}
	form.Set("backend", "cuda")
	if code, body := owner.post("/console/models/"+r.model+"/deployments", form); code != http.StatusSeeOther {
		t.Fatalf("valid deployment %d: %s", code, body)
	}
	if r.scalar("SELECT (spec->'model'->>'repo') || ' ' || mode || ' ' || (selector->>'gpu') FROM deployments WHERE name = 'coder-gpu'") != "example-org/coder on_demand b300" {
		t.Fatal("deployment fields were not stored")
	}
	same := url.Values{"model_kind": {"path"}, "model_path": {"/srv/coder"}, "pool": {"mac"}, "mode": {"warm"}, "min_replicas": {"1"}, "max_replicas": {"2"}}
	if code, page := owner.follow("/console/deployments/coder-mac", same); code != http.StatusSeeOther || !strings.Contains(page, "Deployment saved.") {
		t.Fatalf("placement-only save %d", code)
	}
	same.Set("context", "65536")
	if code, page := owner.follow("/console/deployments/coder-mac", same); code != http.StatusSeeOther || !strings.Contains(page, "revision 2") {
		t.Fatalf("spec change %d", code)
	}
	if r.scalar("SELECT revision::text || ' ' || max_replicas::text FROM deployments WHERE name = 'coder-mac'") != "2 2" {
		t.Fatal("revision did not bump exactly once")
	}
	if code, _ := owner.post("/console/deployments/coder-mac/status", url.Values{"status": {"retired"}}); code != http.StatusUnprocessableEntity {
		t.Fatalf("retire without confirmation answered %d", code)
	}
	if code, _ := owner.post("/console/deployments/coder-mac/status", url.Values{"status": {"retired"}, "confirm": {"on"}}); code != http.StatusSeeOther {
		t.Fatalf("retire %d", code)
	}
	if r.scalar("SELECT desired FROM replicas WHERE id = $1", r.replica) != "stopped" || r.audited("deployment.retired") != 1 {
		t.Fatal("retiring did not stop replicas with an audit event")
	}
	if r.audited("deployment.created") != 1 || r.audited("deployment.updated") != 2 {
		t.Fatal("deployment changes were not audited")
	}
}

func TestNodeControlsAndEnrolment(t *testing.T) {
	r := newRig(t)
	op := r.user("ops@example.com", "fleet_operator", "")
	resp, _ := op.do("POST", "/console/nodes/studio-2/action", url.Values{"action": {"approve"}, "next": {"/console"}})
	if resp.StatusCode != http.StatusSeeOther || resp.Header.Get("Location") != "/console" || r.scalar("SELECT status FROM nodes WHERE name = 'studio-2'") != "active" {
		t.Fatalf("approve %d to %s", resp.StatusCode, resp.Header.Get("Location"))
	}
	resp, _ = op.do("POST", "/console/nodes/studio-1/action", url.Values{"action": {"drain"}, "next": {"https://evil.example/"}})
	if resp.Header.Get("Location") != "/console/nodes" || r.scalar("SELECT desired FROM replicas WHERE id = $1", r.replica) != "stopped" {
		t.Fatalf("drain redirected to %q", resp.Header.Get("Location"))
	}
	if code, _ := op.post("/console/nodes/studio-1/action", url.Values{"action": {"online"}}); code != http.StatusSeeOther {
		t.Fatalf("bring online %d", code)
	}
	if code, _ := op.post("/console/nodes/studio-1/action", url.Values{"action": {"disable"}}); code != http.StatusSeeOther ||
		r.scalar("SELECT status FROM nodes WHERE name = 'studio-1'") != "disabled" {
		t.Fatalf("take offline %d", code)
	}
	if code, _ := op.post("/console/nodes/studio-1/action", url.Values{"action": {"revoke"}}); code != http.StatusUnprocessableEntity {
		t.Fatalf("revoke without confirmation answered %d", code)
	}
	if code, _ := op.post("/console/nodes/studio-1/action", url.Values{"action": {"revoke"}, "confirm": {"on"}}); code != http.StatusSeeOther {
		t.Fatalf("revoke %d", code)
	}
	for _, a := range []string{"node.approve", "node.drain", "node.online", "node.disable", "node.revoke"} {
		if r.audited(a) != 1 {
			t.Errorf("%s was not audited once", a)
		}
	}
	code, body := op.post("/console/nodes/tokens", url.Values{"pool": {"gpu"}, "labels": {"gpu=b300"}, "uses": {"2"}, "ttl_hours": {"4"}})
	want := "starlight-node enroll --server https://starlight.example.com:9443 --token "
	if code != http.StatusOK || !strings.Contains(body, want) || !strings.Contains(body, "--ca-sha256 "+r.c.Trust.Fingerprint()) {
		t.Fatalf("enrolment token page %d: %s", code, body)
	}
	tok := strings.Fields(body[strings.Index(body, want)+len(want):])[0]
	if len(tok) < 20 {
		t.Fatalf("token %q", tok)
	}
	if _, list := op.get("/console/nodes"); strings.Contains(list, tok) || !strings.Contains(list, "gpu") {
		t.Fatal("the enrolment token was shown again, or the token list is missing")
	}
	if r.scalar("SELECT count(*)::text FROM enrollment_tokens WHERE pool = 'gpu' AND max_uses = 2") != "1" || r.audited("node.token_created") != 1 {
		t.Fatal("token not stored or not audited")
	}
}

func TestProviderCredentialIsWriteOnly(t *testing.T) {
	r := newRig(t)
	owner := r.user("owner@example.com", "owner", "")
	const key = "sk-" + "test-0123456789abcdef"
	form := url.Values{"name": {"openai-main"}, "kind": {"openai"}, "credential": {key}, "max_class": {"internal"}, "headers": {"OpenAI-Organization: org-example"},
		"enabled": {"on"}}
	if code, body := owner.post("/console/providers", form); code != http.StatusSeeOther {
		t.Fatalf("create provider %d: %s", code, body)
	}
	if r.scalar("SELECT base_url FROM providers WHERE name = 'openai-main'") != "https://api.openai.com/v1" {
		t.Fatal("the kind's default base URL was not applied")
	}
	if strings.Contains(r.scalar("SELECT encode(credential, 'escape') FROM providers WHERE name = 'openai-main'"), key) {
		t.Fatal("the credential is stored in plain text")
	}
	if _, page := owner.get("/console/providers/openai-main"); strings.Contains(page, key) || !strings.Contains(page, "stored; type a new key to replace it") {
		t.Fatal("the provider page shows the credential or hides that one is stored")
	}
	if strings.Contains(r.scalar("SELECT string_agg(details::text, ' ') FROM audit_events"), key) {
		t.Fatal("the credential reached the audit log")
	}
	form = url.Values{"kind": {"openai"}, "base_url": {"https://api.openai.com/v1"}, "max_class": {"internal"}, "clear_credential": {"on"}}
	if code, _ := owner.post("/console/providers/openai-main", form); code != http.StatusSeeOther {
		t.Fatalf("update provider %d", code)
	}
	if r.scalar("SELECT (credential IS NULL)::text || ' ' || enabled::text FROM providers WHERE name = 'openai-main'") != "true false" {
		t.Fatal("clearing the credential or disabling did not apply")
	}
	if r.audited("provider.created") != 1 || r.audited("provider.updated") != 1 {
		t.Fatal("provider changes were not audited")
	}
}

func TestLimitsSubscriptionsAndUsageExport(t *testing.T) {
	r := newRig(t)
	owner := r.user("owner@example.com", "owner", "")
	if code, body := owner.post("/console/limits", url.Values{"kind": {"cost_per_month"}, "value": {"2500"}, "model": {r.model}}); code != http.StatusSeeOther {
		t.Fatalf("set limit %d: %s", code, body)
	}
	if _, page := owner.get("/console/limits"); !strings.Contains(page, "$2500.00") || !strings.Contains(page, "team/coder") {
		t.Fatal("the limit is not listed in dollars for its model")
	}
	limit := r.scalar("SELECT id::text FROM limits WHERE kind = 'cost_per_month'")
	if code, _ := owner.post("/console/limits/"+limit+"/remove", nil); code != http.StatusSeeOther || r.audited("limit.removed") != 1 {
		t.Fatalf("remove limit %d", code)
	}
	resp, _ := owner.do("POST", "/console/subscriptions", url.Values{"vendor": {"Example AI"}, "product": {"Team seats"}, "seats": {"2"},
		"seat_price": {"30"}, "period": {"month"}})
	sub := strings.TrimPrefix(resp.Header.Get("Location"), "/console/subscriptions/")
	if resp.StatusCode != http.StatusSeeOther || !ids.Valid(sub) {
		t.Fatalf("create subscription %d to %s", resp.StatusCode, resp.Header.Get("Location"))
	}
	if code, body := owner.post("/console/subscriptions/"+sub+"/seats", url.Values{"email": {"owner@example.com"}}); code != http.StatusSeeOther {
		t.Fatalf("assign seat %d: %s", code, body)
	}
	if code, body := owner.post("/console/subscriptions/"+sub+"/seats", url.Values{"email": {"nobody@example.com"}}); code != http.StatusUnprocessableEntity ||
		!strings.Contains(body, "first sign-in") {
		t.Fatalf("unknown seat holder %d", code)
	}
	if _, page := owner.get("/console/subscriptions"); !strings.Contains(page, "$60.00") || !strings.Contains(page, "owner@example.com") {
		t.Fatal("monthly cost or the reclamation report is wrong")
	}
	if code, _ := owner.post("/console/subscriptions/"+sub+"/seats/"+owner.user.ID+"/remove", url.Values{"from": {"report"}}); code != http.StatusSeeOther ||
		r.audited("subscription.seat_unassigned") != 1 {
		t.Fatalf("free seat %d", code)
	}
	r.exec(`INSERT INTO usage_events (id, created_at, request_id, principal_type, principal_id, user_id, project_id, model_id, model_name, route_kind,
		protocol, status, outcome, prompt_tokens, completion_tokens, cost_micros, classification, priority)
		VALUES ($1, now(), 'r1', 'user', $2, $2, $3, $4, '=team/coder', 'replica', 'chat.completions', 200, 'ok', 1000, 250, 1500, 'internal', 'standard')`,
		ids.New(), owner.user.ID, r.project, r.model)
	resp, body := owner.do("GET", "/console/usage.csv?group=model", nil)
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/csv") ||
		!strings.Contains(resp.Header.Get("Content-Disposition"), "attachment") {
		t.Fatalf("csv %d %v", resp.StatusCode, resp.Header)
	}
	if body != "model,requests,prompt_tokens,completion_tokens,cached_tokens,cost_usd\n'=team/coder,1,1000,250,0,0.001500\n" {
		t.Fatalf("csv body %q", body)
	}
	if _, page := owner.get("/console/usage?group=model&project=" + r.project); !strings.Contains(page, "1,250") {
		t.Fatal("the usage report does not total tokens")
	}
	if code, _ := owner.get("/console/usage?project=" + ids.New()); code != http.StatusOK {
		t.Fatalf("an owner filtering by any project answered %d", code)
	}
}
