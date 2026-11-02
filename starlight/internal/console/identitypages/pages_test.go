package identitypages_test

import (
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

func TestEveryPageRendersForAnOwner(t *testing.T) {
	r := newRig(t)
	owner := r.user("olu@example.com", "owner")
	other := r.user("pat@example.com", "")
	gid, err := directory.UpsertGroup(r.ctx, r.pool, nil, directory.FromLocal, directory.GroupRef{ExternalID: "platform", DisplayName: "Platform"})
	if err != nil {
		t.Fatal(err)
	}
	r.exec("INSERT INTO group_members (group_id, user_id) VALUES ($1, $2)", gid, other.user.ID)
	if code, _ := owner.post("/console/settings", url.Values{"org_name": {"Example Corp"}, "audit_retention_days": {"365"}}); code != http.StatusSeeOther {
		t.Fatalf("settings save %d", code)
	}
	pages := map[string]string{
		"/console/account": "My account",
		"/console/sso":     "Add a connection",
		"/console/sso/new?choice=generic_oidc:oidc": "Register Starlight at the provider",
		"/console/users":                  "pat@example.com",
		"/console/users/" + other.user.ID: "Change status",
		"/console/groups":                 "Platform",
		"/console/groups/" + gid:          "Grant to the group",
		"/console/access":                 "1 active owner",
		"/console/audit":                  "settings.updated",
		"/console/audit/1":                "Previous hash",
		"/console/settings":               "Workload CA SHA-256",
	}
	for path, want := range pages {
		code, body := owner.get(path)
		if code != 200 || !strings.Contains(body, want) {
			t.Errorf("%s: %d, missing %q", path, code, want)
		}
	}
}

func TestPermissionsRefuseAndHideControls(t *testing.T) {
	r := newRig(t)
	member := r.user("quinn@example.com", "")
	auditor := r.user("rae@example.com", "auditor")
	target := r.user("sam@example.com", "")
	for _, path := range []string{"/console/sso", "/console/users", "/console/groups", "/console/access", "/console/audit", "/console/settings"} {
		if code, _ := member.get(path); code != http.StatusForbidden {
			t.Errorf("member %s answered %d", path, code)
		}
	}
	if code, _ := member.get("/console/account"); code != 200 {
		t.Fatalf("member account page %d", code)
	}
	checks := []struct {
		path, hidden string
	}{{"/console/sso", "Add a connection"}, {"/console/settings", `name="org_name"`}, {"/console/users/" + target.user.ID, "Change status"}}
	for _, c := range checks {
		code, body := auditor.get(c.path)
		if code != 200 || strings.Contains(body, c.hidden) {
			t.Errorf("auditor %s: %d, control %q visible=%v", c.path, code, c.hidden, strings.Contains(body, c.hidden))
		}
	}
	if code, _ := auditor.get("/console/access"); code != http.StatusForbidden {
		t.Fatalf("auditor opened administrators: %d", code)
	}
	writes := map[string]url.Values{
		"/console/sso":                                 {"preset": {"generic_oidc"}, "protocol": {"oidc"}},
		"/console/settings":                            {"org_name": {"Mine now"}},
		"/console/settings/offline":                    {"offline": {"on"}, "confirm": {"yes"}},
		"/console/users/" + target.user.ID + "/status": {"status": {"suspended"}},
		"/console/users/" + target.user.ID + "/revoke": nil,
		"/console/access":                              {"role": {"auditor"}, "email": {"quinn@example.com"}},
	}
	for path, form := range writes {
		if code, _ := auditor.post(path, form); code != http.StatusForbidden {
			t.Errorf("auditor POST %s answered %d", path, code)
		}
	}
	u, _ := directory.GetUser(r.ctx, r.pool, target.user.ID)
	if u.Status != directory.Active {
		t.Fatal("a refused action still changed the user")
	}
}
