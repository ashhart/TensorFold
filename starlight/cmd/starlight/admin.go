package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/login"
	"github.com/ashhart/TensorFold/starlight/internal/server"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
)

func adminCmd() *cli.Command {
	var email string
	return &cli.Command{Name: "admin", Summary: "Break-glass administration that needs database access", Sub: []*cli.Command{
		{Name: "setup-code", Summary: "Issue a new first-owner setup link while no owner exists", Flags: withConfig,
			Run: func(ctx context.Context, _ *flag.FlagSet) error {
				cfg, pool, err := openDB(ctx)
				if err != nil {
					return err
				}
				defer pool.Close()
				code, err := login.IssueSetupCode(ctx, pool, 24*time.Hour)
				if err != nil {
					return err
				}
				fmt.Printf("%s/auth/setup?code=%s\n", strings.TrimRight(cfg.PublicURL, "/"), code)
				return nil
			}},
		{Name: "recovery", Summary: "Issue a 15-minute sign-in link for an owner when SSO is unavailable", Usage: "--email owner@example.com",
			Flags: func(fs *flag.FlagSet) {
				withConfig(fs)
				fs.StringVar(&email, "email", "", "email of an existing, active owner")
			},
			Run: func(ctx context.Context, _ *flag.FlagSet) error {
				if email == "" {
					return cli.ErrUsage
				}
				cfg, pool, err := openDB(ctx)
				if err != nil {
					return err
				}
				defer pool.Close()
				var userID string
				err = pool.QueryRow(ctx, `SELECT u.id FROM users u JOIN role_bindings rb ON rb.subject_type = 'user' AND rb.subject_id = u.id
					WHERE u.email_lower = lower($1) AND u.status = 'active' AND rb.role = 'owner' AND rb.project_id IS NULL`, email).Scan(&userID)
				if db.NotFound(err) {
					return errors.New("no active owner has that email")
				}
				if err != nil {
					return err
				}
				var tok string
				err = db.Tx(ctx, pool, func(tx pgx.Tx) error {
					var err error
					if tok, err = login.IssueRecoveryCode(ctx, tx, userID, 15*time.Minute); err != nil {
						return err
					}
					e := audit.Event{ActorType: "operator", ActorID: hostUser(), Action: "recovery.issued", TargetType: "user", TargetID: userID}
					return audit.Append(ctx, tx, e)
				})
				if err != nil {
					return err
				}
				fmt.Printf("%s/auth/recover?code=%s\n(valid 15 minutes, single use, audited)\n", strings.TrimRight(cfg.PublicURL, "/"), tok)
				return nil
			}},
	}}
}

func hostUser() string {
	h, _ := os.Hostname()
	return os.Getenv("USER") + "@" + h
}

func keysCmd() *cli.Command {
	return &cli.Command{Name: "keys", Summary: "Rotate encryption and signing keys", Sub: []*cli.Command{
		{Name: "rotate", Summary: "Add a new data key for new seals; old keys keep decrypting", Flags: withConfig,
			Run: func(ctx context.Context, _ *flag.FlagSet) error {
				cfg, pool, err := openDB(ctx)
				if err != nil {
					return err
				}
				defer pool.Close()
				keys, err := server.LoadKeyring(ctx, pool, cfg.Keys)
				if err != nil {
					return err
				}
				id, err := keys.Rotate(ctx, pool)
				if err == nil {
					fmt.Println("active data key:", id)
				}
				return err
			}},
		{Name: "rotate-permits", Summary: "Publish a new permit key to nodes; gateways sign with it two minutes later", Flags: withConfig,
			Run: func(ctx context.Context, _ *flag.FlagSet) error {
				cfg, pool, err := openDB(ctx)
				if err != nil {
					return err
				}
				defer pool.Close()
				keys, err := server.LoadKeyring(ctx, pool, cfg.Keys)
				if err != nil {
					return err
				}
				return trust.RotatePermitKey(ctx, pool, keys)
			}},
		{Name: "rewrap", Summary: "Re-wrap every data key under the current master key so previous master files can be removed",
			Flags: withConfig, Run: func(ctx context.Context, _ *flag.FlagSet) error {
				cfg, pool, err := openDB(ctx)
				if err != nil {
					return err
				}
				defer pool.Close()
				keys, err := server.LoadKeyring(ctx, pool, cfg.Keys)
				if err != nil {
					return err
				}
				n, err := keys.Rewrap(ctx, pool)
				fmt.Printf("re-wrapped %d data keys under the current master key\n", n)
				return err
			}},
	}}
}

func auditCmd() *cli.Command {
	var since string
	return &cli.Command{Name: "audit", Summary: "Verify or export the audit trail", Sub: []*cli.Command{
		{Name: "verify", Summary: "Recompute the audit hash chain", Flags: withConfig,
			Run: func(ctx context.Context, _ *flag.FlagSet) error {
				_, pool, err := openDB(ctx)
				if err != nil {
					return err
				}
				defer pool.Close()
				res, err := audit.Verify(ctx, pool)
				if err != nil {
					return err
				}
				fmt.Printf("verified %d events, seq %d to %d, anchor %x\n", res.Checked, res.From, res.To, res.Anchor)
				return nil
			}},
		{Name: "export", Summary: "Write events as JSON lines to stdout", Usage: "[--since 2026-01-01T00:00:00Z]",
			Flags: func(fs *flag.FlagSet) {
				withConfig(fs)
				fs.StringVar(&since, "since", "", "RFC 3339 start time")
			},
			Run: func(ctx context.Context, _ *flag.FlagSet) error {
				from := time.Unix(0, 0)
				if since != "" {
					t, err := time.Parse(time.RFC3339, since)
					if err != nil {
						return err
					}
					from = t
				}
				_, pool, err := openDB(ctx)
				if err != nil {
					return err
				}
				defer pool.Close()
				rows, err := pool.Query(ctx, `SELECT seq, created_at, actor_type, actor_id, actor_label, action, target_type, target_id, outcome, ip,
					request_id, details, encode(hash, 'hex') FROM audit_events WHERE created_at >= $1 ORDER BY seq`, from)
				if err != nil {
					return err
				}
				defer rows.Close()
				enc := json.NewEncoder(os.Stdout)
				for rows.Next() {
					var r struct {
						Seq                                                          int64
						At                                                           time.Time
						ActorType, ActorID, ActorLabel, Action, TargetType, TargetID string
						Outcome, IP, RequestID                                       string
						Details                                                      json.RawMessage
						Hash                                                         string
					}
					if err := rows.Scan(&r.Seq, &r.At, &r.ActorType, &r.ActorID, &r.ActorLabel, &r.Action, &r.TargetType, &r.TargetID, &r.Outcome,
						&r.IP, &r.RequestID, &r.Details, &r.Hash); err != nil {
						return err
					}
					if err := enc.Encode(r); err != nil {
						return err
					}
				}
				return rows.Err()
			}},
	}}
}
