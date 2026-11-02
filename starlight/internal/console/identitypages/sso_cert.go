package identitypages

import (
	"crypto/sha1"
	"encoding/hex"
	"net/http"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
)

// clientCert is a connection's private_key_jwt certificate as the administrator registers it at the IdP.
type clientCert struct {
	PEM        string
	Subject    string
	Thumbprint string
	NotAfter   time.Time
	JWK        string
}

// certFor describes the connection's certificate; Thumbprint is the SHA-1 hex form Entra shows after an upload.
func certFor(cn connections.Connection) *clientCert {
	o := cn.Config.OIDC
	if o == nil || o.SigningCertificate == "" {
		return nil
	}
	cert, err := connections.ParseCertificate(o.SigningCertificate)
	if err != nil {
		return nil
	}
	sum := sha1.Sum(cert.Raw)
	jwk, _ := oidc.PublicJWK(o)
	return &clientCert{PEM: connections.CertificatePEM(cert), Subject: cert.Subject.CommonName, Thumbprint: strings.ToUpper(hex.EncodeToString(sum[:])),
		NotAfter: cert.NotAfter, JWK: string(jwk)}
}

func (p pages) ssoCert() {
	c := p.c
	c.Handle("GET /console/sso/{slug}/client-certificate", authz.SSORead, func(q *console.Request) error {
		cn, err := p.conn(q)
		if err != nil {
			return err
		}
		cert := certFor(cn)
		if cert == nil {
			return console.Invalid("This connection has no client certificate.")
		}
		h := q.W.Header()
		h.Set("Content-Type", "application/x-pem-file")
		h.Set("Content-Disposition", `attachment; filename="starlight-`+cn.Slug+`.pem"`)
		h.Set("Cache-Control", "no-store")
		h.Set("X-Content-Type-Options", "nosniff")
		q.W.WriteHeader(http.StatusOK)
		_, err = q.W.Write([]byte(cert.PEM))
		return err
	})
	c.Handle("POST /console/sso/{slug}/client-certificate", authz.SSOWrite, func(q *console.Request) error {
		cn, err := p.conn(q)
		if err != nil {
			return err
		}
		saved, err := c.Login.Connections().RotateClientKey(q.Ctx(), cn.ID, p.recorder(q, "sso.client_certificate_generated", nil))
		if err != nil {
			return friendly(err)
		}
		c.Login.Forget(saved.ID)
		return q.Redirect("/console/sso/"+saved.Slug, "New key and certificate generated. Upload the certificate to the provider now.")
	})
}
