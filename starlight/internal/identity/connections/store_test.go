package connections_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func oidcInput(slug string, domains ...string) connections.Input {
	return connections.Input{Slug: slug, Name: "Corp " + slug, Protocol: connections.OIDC, Preset: "generic_oidc", Domains: domains, Enabled: true,
		Config:  connections.Config{OIDC: &connections.OIDCConfig{Issuer: "https://idp.example.com", ClientID: "starlight"}, Claims: connections.ClaimMap{Groups: "groups"}},
		Secrets: connections.Secrets{ClientSecret: "first-secret"}}
}

func TestCreateSealsSecretsAndRoutesDomains(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	c, err := s.Create(ctx, oidcInput("corp", "Example.COM", "example.com"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Domains) != 1 || c.Domains[0] != "example.com" || !c.HasSecret || c.Config.Groups.Strategy != connections.GroupsClaim {
		t.Fatalf("created %+v", c)
	}
	var raw []byte
	if err := pool.QueryRow(ctx, "SELECT secrets FROM idp_connections WHERE id = $1", c.ID).Scan(&raw); err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("first-secret")) {
		t.Fatal("secret stored in plaintext")
	}
	sec, err := s.Secrets(ctx, c)
	if err != nil || sec.ClientSecret != "first-secret" {
		t.Fatalf("secrets = %+v, %v", sec, err)
	}
	got, err := s.ForEmail(ctx, "Ada@Example.com")
	if err != nil || got.ID != c.ID {
		t.Fatalf("ForEmail = %+v, %v", got, err)
	}
	if _, err := s.ForEmail(ctx, "bo@other.example"); !errors.Is(err, connections.ErrNotFound) {
		t.Fatalf("unrouted domain = %v", err)
	}
	if _, err := s.Create(ctx, oidcInput("other", "example.com"), nil); !errors.Is(err, connections.ErrDomainTaken) {
		t.Fatalf("duplicate domain = %v", err)
	}
	if _, err := s.Create(ctx, oidcInput("corp"), nil); !errors.Is(err, connections.ErrSlugTaken) {
		t.Fatalf("duplicate slug = %v", err)
	}
}

func TestUpdateKeepsSecretsLeftBlank(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	c, err := s.Create(ctx, oidcInput("corp"), nil)
	if err != nil {
		t.Fatal(err)
	}
	in := oidcInput("corp")
	in.Secrets = connections.Secrets{}
	in.Name = "Renamed"
	u, err := s.Update(ctx, c.ID, in, nil)
	if err != nil || u.Name != "Renamed" {
		t.Fatalf("update = %+v, %v", u, err)
	}
	sec, _ := s.Secrets(ctx, u)
	if sec.ClientSecret != "first-secret" {
		t.Fatal("blank secret erased the stored one")
	}
}

func TestSAMLConnectionsGetASigningPair(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	in := connections.Input{Slug: "saml", Name: "SAML", Protocol: connections.SAML, Preset: "generic_saml", Enabled: true,
		Config: connections.Config{SAML: &connections.SAMLConfig{MetadataURL: "https://idp.example.com/md", SignRequests: true}}}
	c, err := s.Create(ctx, in, nil)
	if err != nil {
		t.Fatal(err)
	}
	sec, _ := s.Secrets(ctx, c)
	if !strings.Contains(c.Config.SAML.SPCertificate, "BEGIN CERTIFICATE") || !strings.Contains(sec.SAMLKeyPEM, "PRIVATE KEY") {
		t.Fatal("no SP key pair generated")
	}
	in.Name = "SAML renamed"
	u, err := s.Update(ctx, c.ID, in, nil)
	if err != nil || u.Config.SAML.SPCertificate != c.Config.SAML.SPCertificate {
		t.Fatalf("update rotated the SP certificate: %v", err)
	}
}

func TestValidationRejectsUnsafeConfig(t *testing.T) {
	cases := map[string]func(*connections.Input){
		"http issuer": func(in *connections.Input) { in.Config.OIDC.Issuer = "http://idp.example.com" },
		"bad slug":    func(in *connections.Input) { in.Slug = "Bad Slug" },
		"overridden param": func(in *connections.Input) {
			in.Config.OIDC.AuthParams = map[string]string{"redirect_uri": "https://evil.example"}
		},
		"bearer no aud":     func(in *connections.Input) { in.Config.OIDC.AcceptBearer = true },
		"entra no tenant":   func(in *connections.Input) { in.Preset = "entra" },
		"graph wrong idp":   func(in *connections.Input) { in.Config.Groups.Strategy = connections.GroupsEntraGraph },
		"bad domain":        func(in *connections.Input) { in.Domains = []string{"not a domain"} },
		"claim injection":   func(in *connections.Input) { in.Config.Claims.Email = "email; DROP" },
		"google no domains": func(in *connections.Input) { in.Preset = "google" },
	}
	for name, mutate := range cases {
		in := oidcInput("corp")
		mutate(&in)
		if err := in.Validate(); err == nil {
			t.Errorf("%s: validated", name)
		}
	}
	ok := oidcInput("corp")
	ok.Config.OIDC.Issuer = "http://127.0.0.1:5556"
	if err := ok.Validate(); err != nil {
		t.Errorf("loopback http issuer refused: %v", err)
	}
}

func TestPrivateKeyJWTNeedsAKeyThatSurvivesEdits(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	in := oidcInput("pkjwt")
	in.Config.OIDC.AuthMethod = connections.AuthPrivateJWT
	in.Secrets = connections.Secrets{}
	if _, err := s.Create(ctx, in, nil); err == nil {
		t.Fatal("private_key_jwt accepted without a key")
	}
	certPEM, keyPEM, _ := connections.NewClientKeyPair("starlight-pkjwt")
	otherCert, _, _ := connections.NewClientKeyPair("starlight-other")
	in.Secrets.SigningKeyPEM, in.Config.OIDC.SigningCertificate = keyPEM, otherCert
	if _, err := s.Create(ctx, in, nil); err == nil || !strings.Contains(err.Error(), "does not belong") {
		t.Fatalf("mismatched certificate = %v", err)
	}
	in.Config.OIDC.SigningCertificate = certPEM
	c, err := s.Create(ctx, in, nil)
	if err != nil {
		t.Fatal(err)
	}
	var raw []byte
	if err := pool.QueryRow(ctx, "SELECT secrets FROM idp_connections WHERE id = $1", c.ID).Scan(&raw); err != nil || bytes.Contains(raw, []byte("PRIVATE KEY")) {
		t.Fatalf("signing key stored in plaintext: %v", err)
	}
	in.Secrets = connections.Secrets{}
	in.Config.OIDC.SigningCertificate = ""
	u, err := s.Update(ctx, c.ID, in, nil)
	if err != nil {
		t.Fatalf("edit without re-entering the key: %v", err)
	}
	if sec, _ := s.Secrets(ctx, u); sec.SigningKeyPEM != keyPEM || u.Config.OIDC.SigningCertificate != certPEM {
		t.Fatal("an edit that left the key and certificate blank lost them")
	}
	rotated, err := s.RotateClientKey(ctx, c.ID, nil)
	if err != nil {
		t.Fatal(err)
	}
	sec, _ := s.Secrets(ctx, rotated)
	if rotated.Config.OIDC.SigningCertificate == certPEM || sec.SigningKeyPEM == keyPEM {
		t.Fatal("rotation kept the old pair")
	}
	if _, _, err := connections.ClientKey(sec.SigningKeyPEM, rotated.Config.OIDC.SigningCertificate); err != nil {
		t.Fatalf("rotated pair does not match: %v", err)
	}
	if rotated.Config.OIDC.AuthMethod != connections.AuthPrivateJWT || sec.ClientSecret != "" {
		t.Fatalf("rotation changed other settings: %+v", rotated.Config.OIDC)
	}
}

func TestCertificateFieldsKeepOnlyTheCertificate(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	certPEM, keyPEM, _ := connections.NewClientKeyPair("starlight-pkjwt")
	issuer, _, _ := connections.NewClientKeyPair("starlight-issuer")
	pkjwt := func(pasted string) connections.Input {
		in := oidcInput("pkjwt")
		in.Config.OIDC.AuthMethod = connections.AuthPrivateJWT
		in.Config.OIDC.SigningCertificate = pasted
		in.Secrets = connections.Secrets{SigningKeyPEM: keyPEM}
		return in
	}
	for pasted, says := range map[string]string{certPEM + keyPEM: "private key", keyPEM + certPEM: "private key", certPEM + issuer: "CERTIFICATE block"} {
		var ve *connections.ValidationError
		if _, err := s.Create(ctx, pkjwt(pasted), nil); !errors.As(err, &ve) || ve.Field != "signing_certificate" || !strings.Contains(ve.Message, says) {
			t.Fatalf("pasted %.60q... = %v, want a refusal naming the %s", pasted, err, says)
		}
	}
	c, err := s.Create(ctx, pkjwt("subject=CN = starlight-pkjwt\r\n"+strings.ReplaceAll(certPEM, "\n", "\r\n")+"\r\n\r\n"), nil)
	if err != nil || c.Config.OIDC.SigningCertificate != certPEM {
		t.Fatalf("a certificate pasted with text around it was not stored as the bare certificate: %v", err)
	}
	spCert, spKey, _ := connections.NewSPKeyPair("starlight-sp-saml")
	saml := connections.Input{Slug: "saml", Name: "SAML", Protocol: connections.SAML, Preset: "generic_saml", Enabled: true,
		Config:  connections.Config{SAML: &connections.SAMLConfig{MetadataURL: "https://idp.example.com/md", SPCertificate: spCert + spKey}},
		Secrets: connections.Secrets{SAMLKeyPEM: spKey}}
	var ve *connections.ValidationError
	if _, err := s.Create(ctx, saml, nil); !errors.As(err, &ve) || ve.Field != "sp_certificate" || !strings.Contains(ve.Message, "private key") {
		t.Fatalf("SP certificate with its key = %v", err)
	}
	saml.Config.SAML.SPCertificate = spCert + "\n\n"
	if sc, err := s.Create(ctx, saml, nil); err != nil || sc.Config.SAML.SPCertificate != spCert {
		t.Fatalf("SP certificate not stored as the bare certificate: %v", err)
	}
	var config string
	if err := pool.QueryRow(ctx, "SELECT string_agg(config::text, '') FROM idp_connections").Scan(&config); err != nil || strings.Contains(config, "PRIVATE KEY") {
		t.Fatalf("a private key reached the unsealed config column: %v", err)
	}
}

func TestEntraCertificateAuthNeedsTheCertificate(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	in := oidcInput("entra")
	in.Preset = "entra"
	in.Config.Provider.EntraTenantID = "8d0c6f1e-2c43-4c2d-9a33-0b1c2d3e4f50"
	in.Config.OIDC.AuthMethod = connections.AuthPrivateJWT
	certPEM, keyPEM, _ := connections.NewClientKeyPair("starlight-entra")
	in.Secrets = connections.Secrets{SigningKeyPEM: keyPEM}
	if _, err := s.Create(ctx, in, nil); err == nil || !strings.Contains(err.Error(), "signing_certificate") {
		t.Fatalf("Entra certificate auth without a certificate = %v", err)
	}
	in.Config.OIDC.SigningCertificate = certPEM
	if _, err := s.Create(ctx, in, nil); err != nil {
		t.Fatal(err)
	}
	saml := connections.Input{Slug: "saml", Name: "SAML", Protocol: connections.SAML, Preset: "generic_saml", Enabled: true,
		Config: connections.Config{SAML: &connections.SAMLConfig{MetadataURL: "https://idp.example.com/md"}}}
	c, err := s.Create(ctx, saml, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.RotateClientKey(ctx, c.ID, nil); err == nil {
		t.Fatal("generated a client certificate for a SAML connection")
	}
}

func TestClientKeyRejectsWeakKeys(t *testing.T) {
	small, _ := rsa.GenerateKey(rand.Reader, 1024)
	der := x509.MarshalPKCS1PrivateKey(small)
	cases := map[string]string{
		"not pem":    "not a key",
		"rsa 1024":   string(pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: der})),
		"encrypted":  string(pem.EncodeToMemory(&pem.Block{Type: "ENCRYPTED PRIVATE KEY", Bytes: []byte{0, 0, 0}})),
		"public key": "-----BEGIN PUBLIC KEY-----\nAAAA\n-----END PUBLIC KEY-----\n",
	}
	for name, keyPEM := range cases {
		if _, _, err := connections.ClientKey(keyPEM, ""); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	_, keyPEM, _ := connections.NewClientKeyPair("starlight")
	if _, _, err := connections.ClientKey(keyPEM, "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n"); err == nil {
		t.Error("garbage certificate accepted")
	}
}

func TestAFailedAuditRollsTheChangeBack(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	boom := errors.New("audit unavailable")
	fail := func(pgx.Tx, connections.Connection) error { return boom }
	if _, err := s.Create(ctx, oidcInput("corp"), fail); !errors.Is(err, boom) {
		t.Fatalf("create = %v", err)
	}
	if _, err := s.GetBySlug(ctx, "corp"); !errors.Is(err, connections.ErrNotFound) {
		t.Fatal("a connection committed without its audit record")
	}
	c, err := s.Create(ctx, oidcInput("corp"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetEnabled(ctx, c.ID, !c.Enabled, fail); !errors.Is(err, boom) {
		t.Fatalf("set enabled = %v", err)
	}
	var seen string
	if err := s.Delete(ctx, c.ID, func(_ pgx.Tx, cn connections.Connection) error { seen = cn.Slug; return boom }); !errors.Is(err, boom) {
		t.Fatalf("delete = %v", err)
	}
	got, err := s.Get(ctx, c.ID)
	if err != nil || got.Enabled != c.Enabled || seen != "corp" {
		t.Fatalf("after failed audits: %+v %v, recorder saw %q", got, err, seen)
	}
	if err := s.Delete(ctx, ids.New(), nil); !errors.Is(err, connections.ErrNotFound) {
		t.Fatalf("deleting a missing connection = %v", err)
	}
}

func TestDeletingTheLastOwnersConnectionIsRefused(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	c, err := s.Create(ctx, oidcInput("corp", "example.com"), nil)
	if err != nil {
		t.Fatal(err)
	}
	user, group := ids.New(), ids.New()
	for _, q := range []struct {
		sql  string
		args []any
	}{
		{"INSERT INTO users (id, email, display_name, status, source) VALUES ($1, 'lead@example.com', 'Lead', 'active', 'jit')", []any{user}},
		{"INSERT INTO groups (id, connection_id, source, external_id, display_name) VALUES ($1, $2, 'claim', 'admins', 'Admins')", []any{group, c.ID}},
		{"INSERT INTO group_members (group_id, user_id) VALUES ($1, $2)", []any{group, user}},
		{"INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, 'owner', 'group', $2)", []any{ids.New(), group}},
	} {
		if _, err := pool.Exec(ctx, q.sql, q.args...); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Delete(ctx, c.ID, nil); !errors.Is(err, people.ErrLastOwner) {
		t.Fatalf("deleting the connection whose group holds the only owner = %v", err)
	}
	if _, err := s.Get(ctx, c.ID); err != nil {
		t.Fatalf("the refused delete still removed the connection: %v", err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, 'owner', 'user', $2)", ids.New(), user); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete(ctx, c.ID, nil); err != nil {
		t.Fatalf("a direct owner remains, so the delete should go ahead: %v", err)
	}
}

func TestDisablingOrDeletingAConnectionEndsItsSessions(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	c, err := s.Create(ctx, oidcInput("corp", "example.com"), nil)
	if err != nil {
		t.Fatal(err)
	}
	other, err := s.Create(ctx, oidcInput("other", "other.example"), nil)
	if err != nil {
		t.Fatal(err)
	}
	user := ids.New()
	if _, err := pool.Exec(ctx, "INSERT INTO users (id, email, display_name, status, source) VALUES ($1, 'ada@example.com', 'Ada', 'active', 'jit')", user); err != nil {
		t.Fatal(err)
	}
	signIn := func(connectionID string) string {
		id := ids.New()
		if _, err := pool.Exec(ctx, `INSERT INTO sessions (id, token_hash, user_id, connection_id, auth_time, csrf_key, idle_expires_at, expires_at)
			VALUES ($1, $2, $3, $4, now(), '\x00', now() + interval '1 hour', now() + interval '1 hour')`, id, []byte(id), user, connectionID); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `INSERT INTO oauth_codes (code_hash, client_id, user_id, session_id, redirect_uri, code_challenge, scope, expires_at)
			VALUES ($1, 'starlight-connect', $2, $3, 'http://127.0.0.1/cb', 'challenge', 'openid', now() + interval '2 minutes')`, []byte(id), user, id); err != nil {
			t.Fatal(err)
		}
		return id
	}
	ended := func(sessionID string) (reason string, codeSpent bool) {
		if err := pool.QueryRow(ctx, `SELECT CASE WHEN s.revoked_at IS NULL THEN '' ELSE s.revoked_reason END, c.used_at IS NOT NULL
			FROM sessions s JOIN oauth_codes c ON c.session_id = s.id WHERE s.id = $1`, sessionID).Scan(&reason, &codeSpent); err != nil {
			t.Fatal(err)
		}
		return reason, codeSpent
	}
	first, bystander := signIn(c.ID), signIn(other.ID)
	if err := s.SetEnabled(ctx, c.ID, false, nil); err != nil {
		t.Fatal(err)
	}
	if reason, spent := ended(first); reason != "connection_disabled" || !spent {
		t.Fatalf("after disabling: session %q, its OAuth code spent %v", reason, spent)
	}
	if reason, spent := ended(bystander); reason != "" || spent {
		t.Fatal("disabling one connection ended another connection's session")
	}
	if err := s.SetEnabled(ctx, c.ID, true, nil); err != nil {
		t.Fatal(err)
	}
	second := signIn(c.ID)
	boom := errors.New("audit unavailable")
	if err := s.Delete(ctx, c.ID, func(pgx.Tx, connections.Connection) error { return boom }); !errors.Is(err, boom) {
		t.Fatalf("delete = %v", err)
	}
	if reason, _ := ended(second); reason != "" {
		t.Fatal("a delete that rolled back still ended the session")
	}
	if err := s.Delete(ctx, c.ID, nil); err != nil {
		t.Fatal(err)
	}
	if reason, spent := ended(second); reason != "connection_deleted" || !spent {
		t.Fatalf("after deleting: session %q, its OAuth code spent %v", reason, spent)
	}
}

func TestRemovingAConnectionEndsItsUsersOAuthGrants(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	s := connections.NewStore(pool, keyring.ForTest())
	c, err := s.Create(ctx, oidcInput("corp", "example.com"), nil)
	if err != nil {
		t.Fatal(err)
	}
	other, err := s.Create(ctx, oidcInput("other", "other.example"), nil)
	if err != nil {
		t.Fatal(err)
	}
	exec := func(sql string, args ...any) {
		t.Helper()
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatalf("%s: %v", sql, err)
		}
	}
	project, ada, bo := ids.New(), ids.New(), ids.New()
	exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'research', 'Research')", project)
	for user, links := range map[string][]string{ada: {c.ID, other.ID}, bo: {other.ID}} {
		exec("INSERT INTO users (id, email, display_name, source) VALUES ($1, $2, 'Someone', 'jit')", user, user+"@example.com")
		for _, conn := range links {
			exec("INSERT INTO identities (id, user_id, connection_id, subject) VALUES ($1, $2, $3, $4)", ids.New(), user, conn, "sub-"+user)
		}
	}
	grant := func(user string) {
		family := ids.New()
		exec(`INSERT INTO refresh_tokens (id, family_id, token_hash, user_id, client_id, scope, family_expires_at, idle_expires_at)
			VALUES ($1, $2, $3, $4, 'starlight-connect', 'openid offline_access', now() + interval '1 day', now() + interval '1 hour')`, ids.New(), family, []byte(ids.New()), user)
		for _, f := range []*string{&family, nil} {
			exec(`INSERT INTO access_tokens (id, token_hash, user_id, family_id, client_id, scope, expires_at)
				VALUES ($1, $2, $3, $4, 'starlight-connect', 'openid', now() + interval '1 hour')`, ids.New(), []byte(ids.New()), user, f)
		}
		exec(`INSERT INTO device_grants (id, device_code_hash, user_code, client_id, scope, status, user_id, expires_at)
			VALUES ($1, $2, $3, 'starlight-cli', 'openid', 'approved', $4, now() + interval '10 minutes')`, ids.New(), []byte(ids.New()), ids.New(), user)
		exec(`INSERT INTO oauth_codes (code_hash, client_id, user_id, redirect_uri, code_challenge, scope, expires_at)
			VALUES ($1, 'starlight-connect', $2, 'http://127.0.0.1/cb', 'challenge', 'openid', now() + interval '2 minutes')`, []byte(ids.New()), user)
	}
	live := func(user string) int {
		t.Helper()
		var n int
		if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM refresh_tokens WHERE user_id = $1 AND revoked_at IS NULL)
			+ (SELECT count(*) FROM access_tokens WHERE user_id = $1 AND revoked_at IS NULL)
			+ (SELECT count(*) FROM device_grants WHERE user_id = $1 AND status = 'approved')
			+ (SELECT count(*) FROM oauth_codes WHERE user_id = $1 AND used_at IS NULL)`, user).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	key, signedIn, keyed := ids.New(), ids.New(), ids.New()
	exec("INSERT INTO api_keys (id, public_id, secret_hash, name, owner_type, owner_id, project_id) VALUES ($1, $2, $3, 'ci', 'user', $4, $5)",
		key, ids.New(), []byte(ids.New()), ada, project)
	for task, minter := range map[string]*string{signedIn: nil, keyed: &key} {
		exec(`INSERT INTO tasks (id, root_id, depth, name, owner_type, owner_id, project_id, token_hash, models, max_depth, max_concurrency, api_key_id, expires_at)
			VALUES ($1, $1, 0, 'agent', 'user', $2, $3, $4, '{}', 1, 1, $5, now() + interval '1 day')`, task, ada, project, []byte(ids.New()), minter)
	}
	grant(ada)
	grant(bo)
	disabled := oidcInput("corp", "example.com")
	disabled.Enabled = false
	for _, step := range []struct {
		name   string
		remove func() error
	}{
		{"disabled", func() error { return s.SetEnabled(ctx, c.ID, false, nil) }},
		{"saved disabled", func() error {
			if err := s.SetEnabled(ctx, c.ID, true, nil); err != nil {
				return err
			}
			grant(ada)
			_, err := s.Update(ctx, c.ID, disabled, nil)
			return err
		}},
		{"deleted", func() error { grant(ada); return s.Delete(ctx, c.ID, nil) }},
	} {
		if err := step.remove(); err != nil {
			t.Fatalf("%s: %v", step.name, err)
		}
		if n := live(ada); n != 0 {
			t.Fatalf("%s: %d of Ada's OAuth grants still live", step.name, n)
		}
		if step.name == "saved disabled" {
			grant(ada)
			if _, err := s.Update(ctx, c.ID, disabled, nil); err != nil || live(ada) != 5 {
				t.Fatalf("editing a connection that was already disabled ended grants again: %v", err)
			}
		}
	}
	var keys int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM api_keys WHERE revoked_at IS NULL").Scan(&keys); err != nil || keys != 1 || live(bo) != 5 {
		t.Fatalf("live API keys %d (%v), Bo's grants %d: only Ada's OAuth grants should end", keys, err, live(bo))
	}
	var signedInLive, keyedLive bool
	if err := pool.QueryRow(ctx, "SELECT (SELECT revoked_at IS NULL FROM tasks WHERE id = $1), (SELECT revoked_at IS NULL FROM tasks WHERE id = $2)",
		signedIn, keyed).Scan(&signedInLive, &keyedLive); err != nil || signedInLive || !keyedLive {
		t.Fatalf("task minted at sign-in live %v, task minted by an API key live %v (%v)", signedInLive, keyedLive, err)
	}
}
