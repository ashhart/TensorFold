package identitypages_test

import (
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/settings"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

var apiKey = regexp.MustCompile(`slr_[a-z2-7_]+`)

func TestUserStatusAndLastOwnerProtection(t *testing.T) {
	r := newRig(t)
	owner := r.user("vic@example.com", "owner")
	admin := r.user("wren@example.com", "security_admin")
	staff := r.user("xan@example.com", "")
	if code, _ := admin.post("/console/users/"+staff.user.ID+"/status", url.Values{"status": {"suspended"}}); code != http.StatusSeeOther {
		t.Fatalf("suspend %d", code)
	}
	if code, _ := staff.get("/console/account"); code == 200 {
		t.Fatal("suspended user's session still works")
	}
	if code, _ := admin.post("/console/users/"+staff.user.ID+"/status", url.Values{"status": {"deactivated"}}); code != http.StatusUnprocessableEntity {
		t.Fatalf("deactivation without confirmation %d", code)
	}
	if code, _ := admin.post("/console/users/"+owner.user.ID+"/status", url.Values{"status": {"suspended"}}); code != http.StatusForbidden {
		t.Fatalf("security admin suspended an owner: %d", code)
	}
	second := r.user("yara@example.com", "owner")
	if code, _ := owner.post("/console/users/"+second.user.ID+"/status", url.Values{"status": {"suspended"}}); code != http.StatusSeeOther {
		t.Fatalf("owner suspending a second owner %d", code)
	}
	if code, body := second.get("/console/account"); code == 200 {
		t.Fatalf("suspended owner kept a session: %.80s", body)
	}
	if code, _ := owner.post("/console/users/"+owner.user.ID+"/status", url.Values{"status": {"suspended"}}); code != http.StatusUnprocessableEntity {
		t.Fatalf("self-suspension answered %d", code)
	}
	var n int
	_ = r.pool.QueryRow(r.ctx, "SELECT count(*) FROM audit_events WHERE action = 'user.status'").Scan(&n)
	if n != 2 {
		t.Fatalf("status audit events %d", n)
	}
}

func TestBindingsGroupsAndAccess(t *testing.T) {
	r := newRig(t)
	owner := r.user("zed@example.com", "owner")
	admin := r.user("abe@example.com", "security_admin")
	gid, err := directory.UpsertGroup(r.ctx, r.pool, nil, directory.FromLocal, directory.GroupRef{ExternalID: "ml", DisplayName: "ML"})
	if err != nil {
		t.Fatal(err)
	}
	if code, _ := admin.post("/console/groups/"+gid+"/bindings", url.Values{"role": {"member"}, "project": {r.project}}); code != http.StatusSeeOther {
		t.Fatalf("project grant %d", code)
	}
	if code, _ := admin.post("/console/groups/"+gid+"/bindings", url.Values{"role": {"owner"}}); code != http.StatusForbidden {
		t.Fatalf("security admin granted owner %d", code)
	}
	if code, body := admin.post("/console/groups/"+gid+"/bindings", url.Values{"role": {"member"}}); code != http.StatusUnprocessableEntity {
		t.Fatalf("project role at org scope %d %.120s", code, body)
	}
	if code, _ := admin.post("/console/access", url.Values{"role": {"auditor"}, "email": {"zed@example.com"}}); code != http.StatusSeeOther {
		t.Fatalf("access grant %d", code)
	}
	var ownerBinding, auditorBinding string
	_ = r.pool.QueryRow(r.ctx, "SELECT id FROM role_bindings WHERE role = 'owner'").Scan(&ownerBinding)
	_ = r.pool.QueryRow(r.ctx, "SELECT id FROM role_bindings WHERE role = 'auditor'").Scan(&auditorBinding)
	if code, _ := admin.post("/console/access/"+ownerBinding+"/delete", nil); code != http.StatusForbidden {
		t.Fatalf("security admin removed an owner %d", code)
	}
	if code, body := owner.post("/console/access/"+ownerBinding+"/delete", nil); code != http.StatusUnprocessableEntity || !strings.Contains(body, "without an active owner") {
		t.Fatalf("last owner removed: %d", code)
	}
	if code, _ := owner.post("/console/access/"+auditorBinding+"/delete", nil); code != http.StatusSeeOther {
		t.Fatalf("remove auditor %d", code)
	}
	if r.audited("role.granted") != 2 || r.audited("role.revoked") != 1 {
		t.Fatalf("role audit granted=%d revoked=%d", r.audited("role.granted"), r.audited("role.revoked"))
	}
}

func TestAccountKeysDevicesSessionsAndTasks(t *testing.T) {
	r := newRig(t)
	me := r.user("bea@example.com", "")
	r.exec("INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'member', 'user', $2, $3)", ids.New(), me.user.ID, r.project)
	code, body := me.post("/console/account/keys", url.Values{"project": {r.project}, "name": {"laptop"}, "priority": {"standard"}, "expires_days": {"30"}})
	tok := apiKey.FindString(body)
	if code != 200 || tok == "" {
		t.Fatalf("issue %d: no key shown", code)
	}
	if _, body = me.get("/console/account"); strings.Contains(body, tok) {
		t.Fatal("key shown twice")
	}
	var keyID string
	_ = r.pool.QueryRow(r.ctx, "SELECT id FROM api_keys WHERE secret_hash = $1", secret.Hash(tok)).Scan(&keyID)
	code, body = me.post("/console/account/keys/"+keyID+"/rotate", url.Values{"grace_hours": {"1"}})
	if code != 200 || apiKey.FindString(body) == "" || apiKey.FindString(body) == tok {
		t.Fatalf("rotate %d", code)
	}
	other := r.user("cal@example.com", "")
	if code, _ := other.post("/console/account/keys/"+keyID+"/revoke", nil); code != http.StatusNotFound {
		t.Fatalf("revoked someone else's key: %d", code)
	}
	if code, _ := me.post("/console/account/keys", url.Values{"project": {ids.New()}, "name": {"x"}}); code != http.StatusUnprocessableEntity {
		t.Fatalf("key in a foreign project %d", code)
	}
	family, task := ids.New(), ids.New()
	_, hash := secret.Generate(secret.Refresh)
	r.exec(`INSERT INTO refresh_tokens (id, family_id, token_hash, user_id, client_id, scope, device_name, family_expires_at, idle_expires_at)
		VALUES ($1, $2, $3, $4, 'tensorfold-connect', 'inference', 'Laptop', now() + interval '1 day', now() + interval '1 day')`, ids.New(), family, hash, me.user.ID)
	_, th := secret.Generate(secret.Task)
	r.exec(`INSERT INTO tasks (id, root_id, depth, name, owner_type, owner_id, project_id, token_hash, models, max_depth, max_concurrency, expires_at)
		VALUES ($1, $1, 0, 'refactor', 'user', $2, $3, $4, '{qwen}', 1, 2, now() + interval '1 day')`, task, me.user.ID, r.project, th)
	if _, body = me.get("/console/account"); !strings.Contains(body, "Laptop") || !strings.Contains(body, "refactor") {
		t.Fatal("device or task missing from the account page")
	}
	if code, _ := me.post("/console/account/devices/"+family+"/revoke", nil); code != http.StatusSeeOther {
		t.Fatalf("device revoke %d", code)
	}
	if code, _ := me.post("/console/account/tasks/"+task+"/revoke", nil); code != http.StatusSeeOther {
		t.Fatalf("task revoke %d", code)
	}
	var liveFamily, liveTask bool
	_ = r.pool.QueryRow(r.ctx, "SELECT bool_or(revoked_at IS NULL) FROM refresh_tokens WHERE family_id = $1", family).Scan(&liveFamily)
	_ = r.pool.QueryRow(r.ctx, "SELECT revoked_at IS NULL FROM tasks WHERE id = $1", task).Scan(&liveTask)
	if liveFamily || liveTask {
		t.Fatal("device or task survived revocation")
	}
	second := r.signIn(me.user)
	var sid string
	_ = r.pool.QueryRow(r.ctx, "SELECT id FROM sessions WHERE user_id = $1 ORDER BY created_at LIMIT 1", me.user.ID).Scan(&sid)
	if code, _ := second.post("/console/account/sessions/"+sid+"/revoke", nil); code != http.StatusSeeOther {
		t.Fatalf("session revoke %d", code)
	}
	if code, _ := me.get("/console/account"); code == 200 {
		t.Fatal("revoked session still works")
	}
	for _, a := range []string{"key.issued", "key.rotated", "device.revoked", "task.revoked", "session.revoked"} {
		if r.audited(a) != 1 {
			t.Errorf("audit %s = %d", a, r.audited(a))
		}
	}
}

func TestSettingsOfflineConfirmationAndAuditVerify(t *testing.T) {
	r := newRig(t)
	owner := r.user("dot@example.com", "owner")
	code, body := owner.post("/console/settings/offline", url.Values{"offline": {"on"}})
	if code != 200 || !strings.Contains(body, "Turn offline mode on?") {
		t.Fatalf("confirmation step %d", code)
	}
	if s, _ := settings.Get(r.ctx, r.pool); s.Offline {
		t.Fatal("offline mode changed before confirmation")
	}
	if code, _ := owner.post("/console/settings/offline", url.Values{"offline": {"on"}, "confirm": {"yes"}}); code != http.StatusSeeOther {
		t.Fatalf("confirm %d", code)
	}
	if code, body := owner.post("/console/settings", url.Values{"org_name": {"Example Labs"}, "audit_retention_days": {"7"}}); code != http.StatusUnprocessableEntity ||
		!strings.Contains(body, "30 to 3650") {
		t.Fatalf("short retention accepted: %d", code)
	}
	if code, _ := owner.post("/console/settings", url.Values{"org_name": {"Example Labs"}, "audit_retention_days": {"0"}}); code != http.StatusSeeOther {
		t.Fatalf("save %d", code)
	}
	s, _ := settings.Get(r.ctx, r.pool)
	if !s.Offline || s.OrgName != "Example Labs" || s.AuditRetentionDays != 0 || r.audited("settings.offline") != 1 {
		t.Fatalf("settings %+v", s)
	}
	code, body = owner.post("/console/audit/verify", nil)
	if code != 200 || !strings.Contains(body, "every hash matches") {
		t.Fatalf("verify %d", code)
	}
	r.exec("UPDATE audit_events SET details = '{\"tampered\": true}' WHERE seq = 1")
	if _, body = owner.post("/console/audit/verify", nil); !strings.Contains(body, "altered") {
		t.Fatal("tampering not reported")
	}
	since := time.Now().Add(-time.Hour).UTC().Format("2006-01-02")
	if code, body = owner.get("/console/audit?action=settings.&from=" + since); code != 200 || !strings.Contains(body, "settings.offline") || strings.Contains(body, "role.granted") {
		t.Fatalf("filtered audit %d", code)
	}
}
