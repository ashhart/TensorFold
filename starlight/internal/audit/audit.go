// Package audit appends tamper-evident security events, each hashing the one before it.
package audit

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/principal"
)

// chainLock serializes appends so sequence numbers and hashes form one chain.
const chainLock = 7314_0101

// Event is one security-relevant action.
type Event struct {
	ActorType  string         `json:"actor_type"`
	ActorID    string         `json:"actor_id"`
	ActorLabel string         `json:"actor_label"`
	Action     string         `json:"action"`
	TargetType string         `json:"target_type"`
	TargetID   string         `json:"target_id"`
	Outcome    string         `json:"outcome"`
	IP         string         `json:"ip"`
	UserAgent  string         `json:"user_agent"`
	RequestID  string         `json:"request_id"`
	Details    map[string]any `json:"details"`
}

// Record is a stored event with its place in the chain.
type Record struct {
	Seq       int64     `json:"seq"`
	CreatedAt time.Time `json:"created_at"`
	Event
	PrevHash []byte `json:"prev_hash"`
	Hash     []byte `json:"hash"`
}

// Actor fills actor fields from a principal; a nil principal is the system.
func Actor(p *principal.Principal) Event {
	if p == nil {
		return Event{ActorType: "system", ActorID: "starlight"}
	}
	return Event{ActorType: string(p.Type), ActorID: p.ID, ActorLabel: p.Label()}
}

// FromRequest fills actor and client fields for an event caused by r.
func FromRequest(r *http.Request, proxies *httpx.Proxies, p *principal.Principal) Event {
	e := Actor(p)
	if ip := proxies.ClientIP(r); ip.IsValid() {
		e.IP = ip.String()
	}
	e.UserAgent = truncate(r.UserAgent(), 256)
	e.RequestID = httpx.RequestID(r.Context())
	return e
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func digest(prev []byte, seq int64, at time.Time, e Event) []byte {
	if e.Details == nil {
		e.Details = map[string]any{}
	}
	body, _ := json.Marshal(struct {
		Seq int64  `json:"seq"`
		At  string `json:"at"`
		Event
	}{seq, at.UTC().Format(time.RFC3339Nano), e})
	h := sha256.New()
	h.Write(prev)
	h.Write(body)
	return h.Sum(nil)
}

// Append writes e inside tx, chained to the latest event.
func Append(ctx context.Context, tx pgx.Tx, e Event) error {
	if e.Outcome == "" {
		e.Outcome = "success"
	}
	details, err := normalize(e.Details)
	if err != nil {
		return err
	}
	e.Details = details
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", chainLock); err != nil {
		return err
	}
	var seq int64
	prev := make([]byte, 32)
	err = tx.QueryRow(ctx, "SELECT seq, hash FROM audit_events ORDER BY seq DESC LIMIT 1").Scan(&seq, &prev)
	if err != nil && !db.NotFound(err) {
		return err
	}
	seq++
	at := time.Now().UTC().Truncate(time.Microsecond)
	hash := digest(prev, seq, at, e)
	raw, err := json.Marshal(e.Details)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `INSERT INTO audit_events (seq, created_at, actor_type, actor_id, actor_label, action, target_type, target_id,
		outcome, ip, user_agent, request_id, details, prev_hash, hash) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)`,
		seq, at, e.ActorType, e.ActorID, e.ActorLabel, e.Action, e.TargetType, e.TargetID, e.Outcome, e.IP, e.UserAgent, e.RequestID, raw, prev, hash)
	return err
}

// Log appends e in its own transaction.
func Log(ctx context.Context, pool *pgxpool.Pool, e Event) error {
	return db.Tx(ctx, pool, func(tx pgx.Tx) error { return Append(ctx, tx, e) })
}

// Result summarises a chain verification.
type Result struct {
	From, To int64
	Checked  int64
	Anchor   []byte
}

// Verify recomputes every hash from the oldest retained event and fails at the first break.
func Verify(ctx context.Context, q db.Querier) (Result, error) {
	var res Result
	var last int64 = -1
	var prev []byte
	for {
		rows, err := q.Query(ctx, `SELECT seq, created_at, actor_type, actor_id, actor_label, action, target_type, target_id, outcome,
			ip, user_agent, request_id, details, prev_hash, hash FROM audit_events WHERE seq > $1 ORDER BY seq LIMIT 1000`, last)
		if err != nil {
			return res, err
		}
		n := 0
		for rows.Next() {
			var r Record
			var details []byte
			if err := rows.Scan(&r.Seq, &r.CreatedAt, &r.ActorType, &r.ActorID, &r.ActorLabel, &r.Action, &r.TargetType, &r.TargetID,
				&r.Outcome, &r.IP, &r.UserAgent, &r.RequestID, &details, &r.PrevHash, &r.Hash); err != nil {
				rows.Close()
				return res, err
			}
			if r.Details, err = normalizeRaw(details); err != nil {
				rows.Close()
				return res, fmt.Errorf("audit %d: details: %w", r.Seq, err)
			}
			if prev == nil {
				res.From, res.Anchor = r.Seq, r.PrevHash
			} else if r.Seq != last+1 || !bytes.Equal(r.PrevHash, prev) {
				rows.Close()
				return res, fmt.Errorf("audit chain broken at seq %d", r.Seq)
			}
			if !bytes.Equal(digest(r.PrevHash, r.Seq, r.CreatedAt, r.Event), r.Hash) {
				rows.Close()
				return res, fmt.Errorf("audit event %d was altered", r.Seq)
			}
			prev, last = r.Hash, r.Seq
			res.To = r.Seq
			res.Checked++
			n++
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return res, err
		}
		if n < 1000 {
			return res, nil
		}
	}
}

// normalize gives details the exact shape they will have when read back from jsonb.
func normalize(d map[string]any) (map[string]any, error) {
	if d == nil {
		return map[string]any{}, nil
	}
	raw, err := json.Marshal(d)
	if err != nil {
		return nil, err
	}
	return normalizeRaw(raw)
}

func normalizeRaw(raw []byte) (map[string]any, error) {
	out := map[string]any{}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}
