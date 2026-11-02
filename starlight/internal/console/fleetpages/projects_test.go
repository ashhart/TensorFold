package fleetpages_test

import (
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

var secretValue = regexp.MustCompile(`<p class="secret" aria-label="Secret value">([^<]+)</p>`)

func TestProjectAdminsManageOnlyTheirProject(t *testing.T) {
	r := newRig(t)
	r.exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'finance', 'Finance')", ids.New())
	lead := r.user("lead@example.com", "", "project_admin")
	member := r.user("pat@example.com", "", "member")
	r.user("sam@example.com", "", "")
	if code, _ := lead.post("/console/projects/research/members", url.Values{"email": {"sam@example.com"}, "role": {"viewer"}}); code != http.StatusSeeOther {
		t.Fatalf("add member %d", code)
	}
	if code, _ := member.post("/console/projects/research/members", url.Values{"email": {"sam@example.com"}, "role": {"project_admin"}}); code != http.StatusForbidden {
		t.Fatalf("a member adding members answered %d", code)
	}
	if code, _ := lead.get("/console/projects/finance"); code != http.StatusNotFound {
		t.Fatalf("another project answered %d", code)
	}
	if code, _ := lead.post("/console/projects/research/accounts", url.Values{"name": {"ci-runner"}}); code != http.StatusSeeOther {
		t.Fatalf("create service account %d", code)
	}
	account := r.scalar("SELECT id::text FROM service_accounts WHERE name = 'ci-runner'")
	code, page := lead.post("/console/projects/research/accounts/"+account+"/keys", url.Values{"name": {"ci"}, "priority": {"standard"}})
	m := secretValue.FindStringSubmatch(page)
	if code != http.StatusOK || m == nil || len(m[1]) < 20 {
		t.Fatalf("issue key %d: %s", code, page)
	}
	if _, page := lead.get("/console/projects/research"); strings.Contains(page, m[1]) || !strings.Contains(page, "ci-runner") {
		t.Fatal("the key was shown a second time, or the account is missing")
	}
	if code, _ := lead.post("/console/projects/research/limits", url.Values{"scope": {"member"}, "kind": {"requests_per_minute"}, "value": {"30"}}); code != http.StatusSeeOther {
		t.Fatalf("project limit %d", code)
	}
	if code, _ := member.post("/console/projects/research/limits", url.Values{"kind": {"requests_per_minute"}, "value": {"9999"}}); code != http.StatusForbidden {
		t.Fatalf("a member setting limits answered %d", code)
	}
	_, page = member.get("/console/projects/research")
	for _, hidden := range []string{"Add member", "Save limit", "Create service account", "Grant</button>", "Save settings"} {
		if strings.Contains(page, hidden) {
			t.Errorf("a member sees %q", hidden)
		}
	}
	for _, action := range []string{"role.bound", "service_account.created", "limit.set"} {
		if r.audited(action) != 1 {
			t.Errorf("%s was not audited once", action)
		}
	}
	if r.scalar("SELECT scope_type || ' ' || value::text FROM limits") != "member 30" {
		t.Fatal("the member limit was not stored")
	}
}
