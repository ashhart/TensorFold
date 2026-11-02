package login

import (
	"context"
	"errors"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc/oidcfake"
)

func owners(x *harness) int {
	var n int
	if err := x.pool.QueryRow(context.Background(), "SELECT count(*) FROM role_bindings WHERE role = 'owner' AND project_id IS NULL").Scan(&n); err != nil {
		x.t.Fatal(err)
	}
	return n
}

func TestSetupCreatesFirstConnectionAndClaimsOwner(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	code, err := IssueSetupCode(context.Background(), x.pool, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if status, body := x.get(x.client, "/auth/setup?code="+url.QueryEscape(code)); status != 200 || !strings.Contains(body, "Identity provider") {
		t.Fatalf("setup page = %d", status)
	}
	if status, body := x.post(x.client, "/auth/setup", url.Values{"code": {code}, "action": {"choose"}, "choice": {"generic_oidc:oidc"}}); status != 200 ||
		!strings.Contains(body, "Client ID") {
		t.Fatalf("configure step = %d", status)
	}
	form := url.Values{"code": {code}, "action": {"create"}, "preset": {"generic_oidc"}, "protocol": {"oidc"}, "name": {"Corp SSO"},
		"slug": {"corp"}, "issuer": {p.Issuer}, "client_id": {"starlight"}, "client_secret": {"secret"}, "jit": {"on"}, "domains": {"example.com"}}
	if status, body := x.post(x.client, "/auth/setup", form); status != 200 || !strings.Contains(body, "Connection saved") {
		t.Fatalf("create = %d %s", status, body)
	}
	p.SetUser(map[string]any{"sub": "owner-1", "email": "owner@example.com", "email_verified": true})
	if status, body := x.post(x.client, "/auth/setup", url.Values{"code": {code}, "action": {"start"}, "slug": {"corp"}}); status != 200 ||
		body != "hello owner@example.com" {
		t.Fatalf("claim = %d %q", status, body)
	}
	if owners(x) != 1 || x.audits("setup.owner_claimed", "success") != 1 {
		t.Fatal("owner not claimed")
	}
	if status, _ := x.post(x.browser(), "/auth/setup", url.Values{"code": {code}, "action": {"start"}, "slug": {"corp"}}); status != 403 {
		t.Fatalf("setup code reused = %d", status)
	}
	if _, err := IssueSetupCode(context.Background(), x.pool, time.Hour); !errors.Is(err, ErrOwnerExists) {
		t.Fatalf("setup code issued with an owner present: %v", err)
	}
}

func TestRecoveryCodeSignsInOwnerOnce(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	x.oidcConn(p, nil)
	p.SetUser(map[string]any{"sub": "o", "email": "owner@example.com", "email_verified": true})
	x.get(x.client, "/auth/corp/start")
	owner := x.user("owner@example.com")
	if _, err := IssueRecoveryCode(context.Background(), x.pool, owner.ID, time.Hour); err == nil {
		t.Fatal("recovery code issued to a non-owner")
	}
	if _, err := x.pool.Exec(context.Background(), "INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES (gen_random_uuid(), 'owner', 'user', $1)",
		owner.ID); err != nil {
		t.Fatal(err)
	}
	code, err := IssueRecoveryCode(context.Background(), x.pool, owner.ID, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	var expires time.Time
	if err := x.pool.QueryRow(context.Background(), "SELECT expires_at FROM recovery_codes WHERE user_id = $1", owner.ID).Scan(&expires); err != nil {
		t.Fatal(err)
	}
	if time.Until(expires) > MaxRecoveryTTL+time.Minute {
		t.Fatalf("recovery code lives until %v", expires)
	}
	b := x.browser()
	if status, body := x.get(b, "/auth/recover?code="+url.QueryEscape(code)); status != 200 || !strings.Contains(body, "Break-glass") {
		t.Fatalf("recover page = %d", status)
	}
	if status, body := x.post(b, "/auth/recover", url.Values{"code": {code}}); status != 200 || body != "hello owner@example.com" {
		t.Fatalf("recovery sign-in = %d %q", status, body)
	}
	if s := x.session(b); len(s.AMR) != 1 || s.AMR[0] != "recovery" {
		t.Fatalf("recovery session amr %v", s.AMR)
	}
	if status, _ := x.post(x.browser(), "/auth/recover", url.Values{"code": {code}}); status != 403 {
		t.Fatalf("recovery code reused = %d", status)
	}
	if x.audits("auth.recovery_login", "success") != 1 {
		t.Fatal("recovery not audited")
	}
}

func TestCleanupRemovesExpiredState(t *testing.T) {
	x := newHarness(t, 100)
	if err := Cleanup(context.Background(), x.pool); err != nil {
		t.Fatal(err)
	}
}

func TestSetupCodeReopensWhenNoOwnerIsActive(t *testing.T) {
	x := newHarness(t, 60)
	ctx := context.Background()
	if _, err := x.pool.Exec(ctx, `WITH u AS (INSERT INTO users (id, email, display_name, status, source)
		VALUES (gen_random_uuid(), 'gone@example.com', 'Gone', 'suspended', 'jit') RETURNING id)
		INSERT INTO role_bindings (id, role, subject_type, subject_id) SELECT gen_random_uuid(), 'owner', 'user', id FROM u`); err != nil {
		t.Fatal(err)
	}
	if _, err := IssueSetupCode(ctx, x.pool, time.Hour); err != nil {
		t.Fatalf("an owner binding on a suspended account kept break-glass setup closed: %v", err)
	}
}

func TestClaimingOwnerSpendsEverySetupLink(t *testing.T) {
	x := newHarness(t, 100)
	ctx := context.Background()
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	first, err := IssueSetupCode(ctx, x.pool, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	spare, err := IssueSetupCode(ctx, x.pool, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	form := url.Values{"code": {first}, "action": {"create"}, "preset": {"generic_oidc"}, "protocol": {"oidc"}, "name": {"Corp SSO"},
		"slug": {"corp"}, "issuer": {p.Issuer}, "client_id": {"starlight"}, "client_secret": {"secret"}, "jit": {"on"}, "domains": {"example.com"}}
	if status, body := x.post(x.client, "/auth/setup", form); status != 200 || !strings.Contains(body, "Connection saved") {
		t.Fatalf("create = %d", status)
	}
	p.SetUser(map[string]any{"sub": "owner-1", "email": "owner@example.com", "email_verified": true})
	if status, body := x.post(x.client, "/auth/setup", url.Values{"code": {first}, "action": {"start"}, "slug": {"corp"}}); status != 200 ||
		body != "hello owner@example.com" {
		t.Fatalf("claim = %d %q", status, body)
	}
	if n := x.count("SELECT count(*) FROM recovery_codes WHERE purpose = 'setup' AND used_at IS NULL"); n != 0 {
		t.Fatalf("%d setup links still unused after the owner was claimed", n)
	}
	if _, err := x.pool.Exec(ctx, "DELETE FROM role_bindings WHERE role = 'owner'"); err != nil {
		t.Fatal(err)
	}
	if status, _ := x.get(x.browser(), "/auth/setup?code="+url.QueryEscape(spare)); status != 403 {
		t.Fatalf("the spare setup link reopened setup once no owner was active: %d", status)
	}
}
