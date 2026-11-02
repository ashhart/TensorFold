package oidc

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"errors"
	"net/url"
	"time"

	"github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// JWTBearer is the client_assertion_type of an RFC 7523 client assertion.
const JWTBearer = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"

// assertionTTL keeps each client assertion short-lived; the IdP sees it once, straight after it is signed.
const assertionTTL = 3 * time.Minute

// Credential authenticates Starlight at a token endpoint, with a client secret or a key that signs client assertions.
type Credential struct {
	ClientID string
	secret   string
	key      crypto.Signer
	alg      jose.SignatureAlgorithm
	opts     *jose.SignerOptions
	now      func() time.Time
}

// SecretCredential authenticates with a client secret sent in the request body.
func SecretCredential(clientID, clientSecret string) *Credential {
	return &Credential{ClientID: clientID, secret: clientSecret}
}

// KeyCredential signs assertions with keyPEM; with certPEM the header names the certificate by x5t and x5t#S256.
func KeyCredential(clientID, keyPEM, certPEM, keyID string, now func() time.Time) (*Credential, error) {
	key, cert, err := connections.ClientKey(keyPEM, certPEM)
	if err != nil {
		return nil, err
	}
	if now == nil {
		now = time.Now
	}
	c := &Credential{ClientID: clientID, key: key, alg: jose.RS256, now: now}
	if _, ok := key.(*ecdsa.PrivateKey); ok {
		c.alg = jose.ES256
	}
	if keyID == "" {
		keyID = Thumbprint(key.Public())
	}
	c.opts = (&jose.SignerOptions{}).WithType("JWT").WithHeader("kid", keyID)
	if cert != nil {
		s1, s256 := sha1.Sum(cert.Raw), sha256.Sum256(cert.Raw)
		c.opts = c.opts.WithHeader("x5t", b64(s1[:])).WithHeader("x5t#S256", b64(s256[:]))
	}
	return c, nil
}

// Signed reports whether the credential sends client assertions rather than a secret.
func (c *Credential) Signed() bool { return c != nil && c.key != nil }

// Configured reports whether the credential has a secret or a key to authenticate with.
func (c *Credential) Configured() bool {
	return c != nil && c.ClientID != "" && (c.key != nil || c.secret != "")
}

// Assertion signs a client assertion for the token endpoint at audience, with a random jti and a three-minute lifetime.
func (c *Credential) Assertion(audience string) (string, error) {
	if c.key == nil {
		return "", errors.New("oidc: credential has no signing key")
	}
	signer, err := jose.NewSigner(jose.SigningKey{Algorithm: c.alg, Key: c.key}, c.opts)
	if err != nil {
		return "", err
	}
	now := c.now()
	claims := jwt.Claims{
		Issuer:    c.ClientID,
		Subject:   c.ClientID,
		Audience:  jwt.Audience{audience},
		ID:        secret.RandomString(20),
		IssuedAt:  jwt.NewNumericDate(now),
		NotBefore: jwt.NewNumericDate(now),
		Expiry:    jwt.NewNumericDate(now.Add(assertionTTL)),
	}
	return jwt.Signed(signer).Claims(claims).Serialize()
}

// Apply authenticates a token request bound for tokenURL; each call signs a fresh assertion, so none is reused.
func (c *Credential) Apply(form url.Values, tokenURL string) error {
	form.Set("client_id", c.ClientID)
	form.Del("client_secret")
	if c.key == nil {
		form.Set("client_secret", c.secret)
		return nil
	}
	assertion, err := c.Assertion(tokenURL)
	if err != nil {
		return err
	}
	form.Set("client_assertion_type", JWTBearer)
	form.Set("client_assertion", assertion)
	return nil
}

// Thumbprint is the RFC 7638 SHA-256 thumbprint of a public key, Starlight's default key ID.
func Thumbprint(pub crypto.PublicKey) string {
	sum, err := (&jose.JSONWebKey{Key: pub}).Thumbprint(crypto.SHA256)
	if err != nil {
		return ""
	}
	return b64(sum)
}

// PublicJWK is the certificate's public key as a JWK carrying the kid and thumbprints Starlight signs with.
func PublicJWK(o *connections.OIDCConfig) ([]byte, error) {
	if o == nil || o.SigningCertificate == "" {
		return nil, errors.New("oidc: connection has no client certificate")
	}
	cert, err := connections.ParseCertificate(o.SigningCertificate)
	if err != nil {
		return nil, err
	}
	kid, alg := o.SigningKeyID, string(jose.RS256)
	if kid == "" {
		kid = Thumbprint(cert.PublicKey)
	}
	if _, ok := cert.PublicKey.(*ecdsa.PublicKey); ok {
		alg = string(jose.ES256)
	}
	s1, s256 := sha1.Sum(cert.Raw), sha256.Sum256(cert.Raw)
	k := jose.JSONWebKey{Key: cert.PublicKey, KeyID: kid, Algorithm: alg, Use: "sig", Certificates: []*x509.Certificate{cert},
		CertificateThumbprintSHA1: s1[:], CertificateThumbprintSHA256: s256[:]}
	return k.MarshalJSON()
}

func b64(b []byte) string { return base64.RawURLEncoding.EncodeToString(b) }

func urlQuery(s string) string { return url.QueryEscape(s) }
