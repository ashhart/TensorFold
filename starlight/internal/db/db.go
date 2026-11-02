// Package db owns the Postgres pool, transactions and schema migrations.
package db

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Querier is the subset of pgx shared by pools and transactions.
type Querier interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// Options tunes the pool; zero values take safe defaults.
type Options struct {
	MaxConns         int32
	MinConns         int32
	StatementTimeout time.Duration
	ApplicationName  string
}

// opening bounds how many connections one process may have waiting on the server's listen backlog at once.
const opening = 8

// CallTimeout bounds one database call on the request path, so a database that stops answering is not waited on.
const CallTimeout = 10 * time.Second

// Open connects, applies pool limits and verifies the server answers.
func Open(ctx context.Context, url string, o Options) (*pgxpool.Pool, error) {
	cfg, err := poolConfig(url, o)
	if err != nil {
		return nil, err
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	pctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := pool.Ping(pctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("database unreachable: %w", err)
	}
	return pool, nil
}

// poolConfig parses url and applies pool limits, session settings, and dial and connect timeouts.
func poolConfig(url string, o Options) (*pgxpool.Config, error) {
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		return nil, fmt.Errorf("database url: %w", err)
	}
	cfg.MaxConns = 32
	if o.MaxConns > 0 {
		cfg.MaxConns = o.MaxConns
	}
	cfg.MinConns = min(o.MinConns, cfg.MaxConns)
	cfg.MaxConnIdleTime = 5 * time.Minute
	cfg.HealthCheckPeriod = 30 * time.Second
	if cfg.ConnConfig.ConnectTimeout == 0 {
		cfg.ConnConfig.ConnectTimeout = 10 * time.Second
	}
	cfg.ConnConfig.DialFunc = (&dialer{Dialer: net.Dialer{Timeout: 5 * time.Second, KeepAlive: 5 * time.Minute}, slots: make(chan struct{}, opening)}).dial
	timeout := 15 * time.Second
	if o.StatementTimeout > 0 {
		timeout = o.StatementTimeout
	}
	name := o.ApplicationName
	if name == "" {
		name = "tensorfold-starlight"
	}
	cfg.ConnConfig.RuntimeParams["application_name"] = name
	cfg.ConnConfig.RuntimeParams["statement_timeout"] = fmt.Sprint(timeout.Milliseconds())
	cfg.ConnConfig.RuntimeParams["idle_in_transaction_session_timeout"] = "60000"
	cfg.ConnConfig.RuntimeParams["timezone"] = "UTC"
	return cfg, nil
}

// Tx runs fn in a transaction and retries serialization failures and deadlocks with backoff.
func Tx(ctx context.Context, pool *pgxpool.Pool, fn func(pgx.Tx) error) error {
	var err error
	for attempt := 0; attempt < 5; attempt++ {
		err = pgx.BeginFunc(ctx, pool, fn)
		if !Retryable(err) {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt+1) * 7 * time.Millisecond):
		}
	}
	return err
}

// Retryable reports whether err is a transient serialization or deadlock failure.
func Retryable(err error) bool {
	var pg *pgconn.PgError
	if !errors.As(err, &pg) {
		return false
	}
	return pg.Code == "40001" || pg.Code == "40P01"
}

// IsUnique reports whether err is a unique-constraint violation, optionally on a named constraint.
func IsUnique(err error, constraint string) bool {
	var pg *pgconn.PgError
	if !errors.As(err, &pg) || pg.Code != "23505" {
		return false
	}
	return constraint == "" || pg.ConstraintName == constraint
}

// IsForeignKey reports whether err is a foreign-key violation.
func IsForeignKey(err error) bool {
	var pg *pgconn.PgError
	return errors.As(err, &pg) && pg.Code == "23503"
}

// NotFound reports whether err means no row matched.
func NotFound(err error) bool {
	return errors.Is(err, pgx.ErrNoRows)
}

// dialer staggers new connections so a burst never overflows the server's listen backlog, which resets the overflow.
type dialer struct {
	net.Dialer
	slots chan struct{}
}

func (d *dialer) dial(ctx context.Context, network, addr string) (net.Conn, error) {
	select {
	case d.slots <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	var c net.Conn
	var err error
	for try := 0; try < 3; try++ {
		if c, err = d.DialContext(ctx, network, addr); err == nil || !errors.Is(err, syscall.ECONNRESET) && !errors.Is(err, syscall.ECONNREFUSED) {
			break
		}
		time.Sleep(time.Duration(try+1) * 25 * time.Millisecond)
	}
	if err != nil {
		<-d.slots
		return nil, err
	}
	return &opened{Conn: c, release: func() { <-d.slots }}, nil
}

// opened gives its dial slot back when the server first answers, so the slot covers the wait to be accepted.
type opened struct {
	net.Conn
	once    sync.Once
	release func()
}

func (c *opened) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	c.once.Do(c.release)
	return n, err
}

func (c *opened) Close() error {
	c.once.Do(c.release)
	return c.Conn.Close()
}

// Unavailable reports a dependency unreachable, too slow, dropping its connection or refusing work: a retryable 503.
func Unavailable(err error) bool {
	var ce *pgconn.ConnectError
	var pg *pgconn.PgError
	var ne net.Error
	switch {
	case errors.As(err, &ce), pgconn.Timeout(err), errors.Is(err, context.DeadlineExceeded), errors.Is(err, pgconn.ErrConnClosed),
		errors.Is(err, io.ErrUnexpectedEOF), errors.Is(err, syscall.ECONNRESET), errors.Is(err, syscall.EPIPE), errors.As(err, &ne):
		return true
	case errors.As(err, &pg):
		return strings.HasPrefix(pg.Code, "08") || slices.Contains([]string{"57P01", "57P02", "57P03", "57014", "53300", "40P01"}, pg.Code)
	}
	return false
}
