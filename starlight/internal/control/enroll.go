package control

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/ratelimit"
)

var nodeName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$`)

// TokenSpec describes an enrolment token.
type TokenSpec struct {
	Pool        string
	Labels      map[string]string
	MaxClass    string
	AutoApprove bool
	MaxUses     int
	TTL         time.Duration
	CreatedBy   string
}

// CreateToken mints an enrolment token; only its hash is stored.
func CreateToken(ctx context.Context, q db.Querier, s TokenSpec) (string, string, error) {
	if s.Pool == "" {
		s.Pool = "default"
	}
	if s.MaxClass == "" {
		s.MaxClass = "confidential"
	}
	if s.MaxUses <= 0 {
		s.MaxUses = 1
	}
	if s.TTL <= 0 || s.TTL > 30*24*time.Hour {
		s.TTL = 24 * time.Hour
	}
	labels, _ := json.Marshal(s.Labels)
	if s.Labels == nil {
		labels = []byte("{}")
	}
	tok, hash := secret.Generate(secret.Enrollment)
	id := ids.New()
	var by *string
	if ids.Valid(s.CreatedBy) {
		by = &s.CreatedBy
	}
	_, err := q.Exec(ctx, `INSERT INTO enrollment_tokens (id, token_hash, pool, labels, max_classification, auto_approve, max_uses, expires_at, created_by)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`, id, hash, s.Pool, labels, s.MaxClass, s.AutoApprove, s.MaxUses, time.Now().Add(s.TTL), by)
	return tok, id, err
}

func validAddress(a string) bool {
	host, port, err := net.SplitHostPort(a)
	if err != nil || host == "" {
		return false
	}
	n, err := strconv.Atoi(port)
	return err == nil && n > 0 && n < 65536
}

func (s *Server) enroll(w http.ResponseWriter, r *http.Request) {
	if !s.enrollR.Allow(ratelimit.Client(s.Proxies.ClientIP(r))) {
		httpx.Write(w, httpx.Plain, httpx.Errorf(http.StatusTooManyRequests, "slow_down", "too many enrolment attempts"))
		return
	}
	var req wire.EnrollRequest
	if err := decode(r, &req); err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	if k, ok := secret.Parse(req.Token); !ok || k != secret.Enrollment {
		httpx.Write(w, httpx.Plain, httpx.Errorf(http.StatusUnauthorized, "invalid_token", "the enrolment token is not valid"))
		return
	}
	if !nodeName.MatchString(req.Name) || !validAddress(req.Address) {
		httpx.Write(w, httpx.Plain, httpx.Errorf(http.StatusBadRequest, "invalid_node", "name must be a hostname-like label and address host:port"))
		return
	}
	facts, _ := json.Marshal(req.Facts)
	nodeID := ids.New()
	var resp wire.EnrollResponse
	err := db.Tx(r.Context(), s.Pool, func(tx pgx.Tx) error {
		var tokID, pool, maxClass string
		var labels []byte
		var auto bool
		err := tx.QueryRow(r.Context(), `UPDATE enrollment_tokens SET uses = uses + 1
			WHERE token_hash = $1 AND revoked_at IS NULL AND expires_at > now() AND uses < max_uses
			RETURNING id, pool, labels, max_classification, auto_approve`, secret.Hash(req.Token)).Scan(&tokID, &pool, &labels, &maxClass, &auto)
		if db.NotFound(err) {
			return httpx.Errorf(http.StatusUnauthorized, "invalid_token", "the enrolment token is not valid")
		}
		if err != nil {
			return err
		}
		status := "pending"
		if auto {
			status = "active"
		}
		_, err = tx.Exec(r.Context(), `INSERT INTO nodes (id, name, status, pool, labels, max_classification, address, facts, agent_version, last_seen_at,
			approved_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, now(), CASE WHEN $3 = 'active' THEN now() END)`,
			nodeID, req.Name, status, pool, labels, maxClass, req.Address, facts, req.Facts.AgentVersion)
		if db.IsUnique(err, "") {
			return httpx.Errorf(http.StatusConflict, "name_taken", "a node with this name is already enrolled; revoke it or choose another name")
		}
		if err != nil {
			return err
		}
		certPEM, cert, err := s.Trust.CA().SignCSR(pki.KindNode, nodeID, []byte(req.CSR), NodeCertValidity)
		if err != nil {
			return httpx.Errorf(http.StatusBadRequest, "invalid_csr", "the certificate request is not valid")
		}
		if err := recordCert(r.Context(), tx, nodeID, cert.SerialNumber.Text(16), cert.NotAfter); err != nil {
			return err
		}
		e := audit.FromRequest(r, s.Proxies, nil)
		e.ActorType, e.ActorID, e.Action, e.TargetType, e.TargetID = "node", nodeID, "node.enrolled", "node", nodeID
		e.Details = map[string]any{"name": req.Name, "pool": pool, "status": status, "token_id": tokID, "platform": req.Facts.Platform}
		if err := audit.Append(r.Context(), tx, e); err != nil {
			return err
		}
		resp = wire.EnrollResponse{NodeID: nodeID, Status: status, Certificate: string(certPEM), CA: string(s.Trust.CA().CertPEM),
			TrustDomain: s.Trust.TrustDomain, PermitKeys: s.Trust.PublicKeys()}
		return nil
	})
	if err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	httpx.JSON(w, http.StatusCreated, resp)
}

func recordCert(ctx context.Context, q db.Querier, nodeID, serial string, notAfter time.Time) error {
	if _, err := q.Exec(ctx, "INSERT INTO node_certificates (serial, node_id, not_after) VALUES ($1, $2, $3)", serial, nodeID, notAfter); err != nil {
		return err
	}
	_, err := q.Exec(ctx, "UPDATE nodes SET cert_serial = $2, cert_not_after = $3, updated_at = now() WHERE id = $1", nodeID, serial, notAfter)
	return err
}

func (s *Server) renew(w http.ResponseWriter, r *http.Request) {
	nodeID, _, revoked, err := s.node(r)
	if err != nil || revoked {
		httpx.Write(w, httpx.Plain, errNode)
		return
	}
	var req wire.RenewRequest
	if err := decode(r, &req); err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	certPEM, cert, err := s.Trust.CA().SignCSR(pki.KindNode, nodeID, []byte(req.CSR), NodeCertValidity)
	if err != nil {
		httpx.Write(w, httpx.Plain, httpx.Errorf(http.StatusBadRequest, "invalid_csr", "the certificate request is not valid"))
		return
	}
	if err := recordCert(r.Context(), s.Pool, nodeID, cert.SerialNumber.Text(16), cert.NotAfter); err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	httpx.JSON(w, http.StatusOK, wire.RenewResponse{Certificate: string(certPEM)})
}
