package oidc_test

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc/oidcfake"
)

func keyPair(t *testing.T) (certPEM, keyPEM string, cert *x509.Certificate) {
	t.Helper()
	certPEM, keyPEM, err := connections.NewClientKeyPair("starlight-corp")
	if err != nil {
		t.Fatal(err)
	}
	cert, err = connections.ParseCertificate(certPEM)
	if err != nil {
		t.Fatal(err)
	}
	return certPEM, keyPEM, cert
}

func parse(t *testing.T, raw string, key any) (jose.Header, jwt.Claims) {
	t.Helper()
	tok, err := jwt.ParseSigned(raw, []jose.SignatureAlgorithm{jose.RS256, jose.ES256})
	if err != nil {
		t.Fatal(err)
	}
	var c jwt.Claims
	if err := tok.Claims(key, &c); err != nil {
		t.Fatal(err)
	}
	return tok.Headers[0], c
}

func TestClientAssertionHeadersAndClaims(t *testing.T) {
	certPEM, keyPEM, cert := keyPair(t)
	cred, err := oidc.KeyCredential("starlight", keyPEM, certPEM, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	before := time.Now().Add(-time.Second)
	first, err := cred.Assertion("https://idp.example.com/oauth2/token")
	if err != nil {
		t.Fatal(err)
	}
	h, c := parse(t, first, cert.PublicKey)
	s1, s256 := sha1.Sum(cert.Raw), sha256.Sum256(cert.Raw)
	if h.Algorithm != "RS256" || h.KeyID != oidc.Thumbprint(cert.PublicKey) || h.ExtraHeaders["typ"] != "JWT" ||
		h.ExtraHeaders["x5t"] != base64.RawURLEncoding.EncodeToString(s1[:]) || h.ExtraHeaders["x5t#S256"] != base64.RawURLEncoding.EncodeToString(s256[:]) {
		t.Fatalf("header %+v", h)
	}
	if c.Issuer != "starlight" || c.Subject != "starlight" || len(c.Audience) != 1 || c.Audience[0] != "https://idp.example.com/oauth2/token" || len(c.ID) < 20 {
		t.Fatalf("claims %+v", c)
	}
	iat, nbf, exp := c.IssuedAt.Time(), c.NotBefore.Time(), c.Expiry.Time()
	if iat.Before(before) || !nbf.Equal(iat) || exp.Sub(iat) != oidc.AssertionTTLForTest || exp.Sub(iat) > 5*time.Minute {
		t.Fatalf("iat %v nbf %v exp %v", iat, nbf, exp)
	}
	second, _ := cred.Assertion("https://idp.example.com/oauth2/token")
	if _, c2 := parse(t, second, cert.PublicKey); c2.ID == c.ID {
		t.Fatal("two assertions share a jti")
	}
	named, _ := oidc.KeyCredential("starlight", keyPEM, certPEM, "okta-kid-7", nil)
	if raw, _ := named.Assertion("https://idp.example.com/token"); !strings.Contains(raw, ".") {
		t.Fatal("no assertion")
	} else if h, _ := parse(t, raw, cert.PublicKey); h.KeyID != "okta-kid-7" {
		t.Fatalf("configured kid ignored: %q", h.KeyID)
	}
}

func TestClientAssertionWithECKey(t *testing.T) {
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	der, _ := x509.MarshalPKCS8PrivateKey(key)
	keyPEM := string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der}))
	cred, err := oidc.KeyCredential("starlight", keyPEM, "", "kid-1", nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := cred.Assertion("https://idp.example.com/token")
	if err != nil {
		t.Fatal(err)
	}
	if h, _ := parse(t, raw, &key.PublicKey); h.Algorithm != "ES256" || h.KeyID != "kid-1" || h.ExtraHeaders["x5t"] != nil {
		t.Fatalf("header %+v", h)
	}
	p384, _ := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	der, _ = x509.MarshalPKCS8PrivateKey(p384)
	if _, err := oidc.KeyCredential("starlight", string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})), "", "", nil); err == nil {
		t.Fatal("accepted a P-384 key for ES256")
	}
	certPEM, _, _ := keyPair(t)
	if _, err := oidc.KeyCredential("starlight", keyPEM, certPEM, "", nil); err == nil {
		t.Fatal("accepted a certificate that belongs to another key")
	}
}

func certConn(p *oidcfake.Provider, certPEM string) connections.Connection {
	c := conn(p)
	c.Config.OIDC.AuthMethod, c.Config.OIDC.SigningCertificate = connections.AuthPrivateJWT, certPEM
	return c
}

func TestPrivateKeyJWTCodeExchange(t *testing.T) {
	p := oidcfake.New("starlight", "unused-secret")
	defer p.Close()
	certPEM, keyPEM, _ := keyPair(t)
	if err := p.RequireCertificate(certPEM); err != nil {
		t.Fatal(err)
	}
	p.SetUser(map[string]any{"sub": "u1", "email": "ada@example.com"})
	rp := oidc.NewRP(p.Server.Client(), nil)
	sec := connections.Secrets{SigningKeyPEM: keyPEM, ClientSecret: "stale-secret-left-in-storage"}
	res, err := signInWith(t, p, rp, certConn(p, certPEM), sec, "n-1", 0)
	if err != nil || res.Subject != "u1" {
		t.Fatalf("certificate sign-in = %+v, %v", res, err)
	}
	got := p.Assertions()
	if len(got) != 1 || got[0].Grant != "authorization_code" || got[0].Audience != p.Issuer+"/token" || got[0].Alg != "RS256" {
		t.Fatalf("assertions %+v", got)
	}
	if _, err := signIn(t, p, conn(p), "n-1", 0); err == nil || !strings.Contains(err.Error(), "invalid_client") {
		t.Fatalf("client secret accepted where the IdP requires a certificate: %v", err)
	}
}

func TestPrivateKeyJWTRefusals(t *testing.T) {
	p := oidcfake.New("starlight", "unused-secret")
	defer p.Close()
	certPEM, keyPEM, _ := keyPair(t)
	if err := p.RequireCertificate(certPEM); err != nil {
		t.Fatal(err)
	}
	otherCert, otherKey, _ := keyPair(t)
	if _, err := signInWith(t, p, oidc.NewRP(p.Server.Client(), nil), certConn(p, otherCert), connections.Secrets{SigningKeyPEM: otherKey},
		"n-1", 0); err == nil {
		t.Fatal("assertion signed with an unregistered certificate accepted")
	}
	late := oidc.NewRP(p.Server.Client(), func() time.Time { return time.Now().Add(-10 * time.Minute) })
	if _, err := signInWith(t, p, late, certConn(p, certPEM), connections.Secrets{SigningKeyPEM: keyPEM}, "n-1", 0); err == nil ||
		!strings.Contains(err.Error(), "stale") {
		t.Fatalf("stale assertion = %v", err)
	}
	cred, _ := oidc.KeyCredential("starlight", keyPEM, certPEM, "", nil)
	form := url.Values{"grant_type": {"client_credentials"}}
	if err := cred.Apply(form, p.Issuer+"/token"); err != nil {
		t.Fatal(err)
	}
	post := func() int {
		resp, err := p.Server.Client().PostForm(p.Issuer+"/token", form)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if code := post(); code != http.StatusOK {
		t.Fatalf("client credentials with a fresh assertion = %d", code)
	}
	if code := post(); code != http.StatusUnauthorized {
		t.Fatalf("replayed assertion = %d", code)
	}
	if len(p.Assertions()) != 1 {
		t.Fatalf("assertions %+v", p.Assertions())
	}
}

func TestPublicJWKMatchesTheAssertionHeader(t *testing.T) {
	certPEM, keyPEM, cert := keyPair(t)
	o := &connections.OIDCConfig{SigningCertificate: certPEM}
	raw, err := oidc.PublicJWK(o)
	if err != nil {
		t.Fatal(err)
	}
	var k jose.JSONWebKey
	if err := json.Unmarshal(raw, &k); err != nil || !k.IsPublic() || len(k.Certificates) != 1 || k.Use != "sig" || k.Algorithm != "RS256" {
		t.Fatalf("jwk %s: %v", raw, err)
	}
	cred, _ := oidc.KeyCredential("starlight", keyPEM, certPEM, "", nil)
	assertion, _ := cred.Assertion("https://idp.example.com/token")
	if h, _ := parse(t, assertion, cert.PublicKey); h.KeyID != k.KeyID {
		t.Fatalf("jwk kid %q, assertion kid %q", k.KeyID, h.KeyID)
	}
	if strings.Contains(string(raw), "PRIVATE") || strings.Contains(string(raw), `"d":`) {
		t.Fatal("jwk carries private key material")
	}
}
