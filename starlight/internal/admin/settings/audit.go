package settings

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// AuditFilter narrows the audit list; Before pages backwards by sequence number.
type AuditFilter struct {
	Action string
	Actor  string
	Target string
	From   *time.Time
	To     *time.Time
	Before int64
	Limit  int
}

const auditCols = `seq, created_at, actor_type, actor_id, actor_label, action, target_type, target_id, outcome, ip, user_agent, request_id,
	details, prev_hash, hash`

func scanRecord(row interface{ Scan(...any) error }) (audit.Record, error) {
	var r audit.Record
	var details []byte
	err := row.Scan(&r.Seq, &r.CreatedAt, &r.ActorType, &r.ActorID, &r.ActorLabel, &r.Action, &r.TargetType, &r.TargetID, &r.Outcome, &r.IP,
		&r.UserAgent, &r.RequestID, &details, &r.PrevHash, &r.Hash)
	if err != nil {
		return r, err
	}
	_ = json.Unmarshal(details, &r.Details)
	return r, nil
}

func like(s string) string {
	return strings.NewReplacer(`\`, `\\`, "%", `\%`, "_", `\_`).Replace(strings.TrimSpace(s))
}

// AuditEvents returns one page of events, newest first.
func AuditEvents(ctx context.Context, q db.Querier, f AuditFilter) ([]audit.Record, error) {
	if f.Limit <= 0 || f.Limit > 200 {
		f.Limit = 50
	}
	before := f.Before
	if before <= 0 {
		before = 1 << 62
	}
	rows, err := q.Query(ctx, `SELECT `+auditCols+` FROM audit_events
		WHERE seq < $1 AND ($2 = '' OR action LIKE $2 || '%')
		AND ($3 = '' OR actor_id = $3 OR actor_label ILIKE '%' || $3 || '%')
		AND ($4 = '' OR target_id = $4 OR target_type = $4)
		AND ($5::timestamptz IS NULL OR created_at >= $5) AND ($6::timestamptz IS NULL OR created_at < $6)
		ORDER BY seq DESC LIMIT $7`, before, like(f.Action), like(f.Actor), strings.TrimSpace(f.Target), f.From, f.To, f.Limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []audit.Record
	for rows.Next() {
		r, err := scanRecord(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// AuditEvent loads one event by sequence number.
func AuditEvent(ctx context.Context, q db.Querier, seq int64) (audit.Record, error) {
	return scanRecord(q.QueryRow(ctx, "SELECT "+auditCols+" FROM audit_events WHERE seq = $1", seq))
}
