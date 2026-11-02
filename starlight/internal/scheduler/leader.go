// Package scheduler runs the fleet controller on one Starlight instance at a time under a fenced Postgres lease.
package scheduler

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// ErrFenced means a newer leader exists; the stale one must stop writing.
var ErrFenced = errors.New("scheduler: lost leadership")

const leaseTTL = "15 seconds"

// Elector competes for the controller lease.
type Elector struct {
	Pool     *pgxpool.Pool
	Instance string
}

// acquire takes or renews the lease and returns the epoch, which only changes when leadership changes hands.
func (e *Elector) acquire(ctx context.Context) (int64, bool, error) {
	var epoch int64
	err := e.Pool.QueryRow(ctx, `UPDATE controller_leader SET
		epoch = CASE WHEN holder = $1 AND expires_at > now() THEN epoch ELSE epoch + 1 END,
		holder = $1, expires_at = now() + interval '`+leaseTTL+`'
		WHERE id AND (holder = $1 OR holder IS NULL OR expires_at < now()) RETURNING epoch`, e.Instance).Scan(&epoch)
	if db.NotFound(err) {
		return 0, false, nil
	}
	return epoch, err == nil, err
}

// Run calls lead with a context that is cancelled as soon as the lease cannot be renewed.
func (e *Elector) Run(ctx context.Context, lead func(ctx context.Context, epoch int64)) {
	for ctx.Err() == nil {
		epoch, ok, _ := e.acquire(ctx)
		if !ok {
			sleep(ctx, 5*time.Second)
			continue
		}
		lctx, cancel := context.WithCancel(ctx)
		done := make(chan struct{})
		go func() {
			defer close(done)
			lead(lctx, epoch)
		}()
		for lctx.Err() == nil {
			sleep(lctx, 5*time.Second)
			if next, ok, err := e.acquire(ctx); err != nil || !ok || next != epoch {
				cancel()
			}
		}
		cancel()
		<-done
	}
	_, _ = e.Pool.Exec(context.Background(), "UPDATE controller_leader SET expires_at = now() WHERE holder = $1", e.Instance)
}

func sleep(ctx context.Context, d time.Duration) {
	select {
	case <-ctx.Done():
	case <-time.After(d):
	}
}

// Fence aborts tx unless this instance still holds the lease at epoch.
func Fence(ctx context.Context, tx pgx.Tx, instance string, epoch int64) error {
	var ok bool
	err := tx.QueryRow(ctx, `SELECT true FROM controller_leader WHERE id AND holder = $1 AND epoch = $2 AND expires_at > now() FOR SHARE`,
		instance, epoch).Scan(&ok)
	if db.NotFound(err) {
		return ErrFenced
	}
	return err
}
