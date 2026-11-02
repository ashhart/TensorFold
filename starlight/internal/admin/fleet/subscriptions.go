package fleet

import (
	"context"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Subscription is a seat-based frontier product the organisation pays for, such as an assistant or coding tool licence.
type Subscription struct {
	ID              string
	Vendor          string
	Product         string
	Seats           int
	SeatPriceMicros int64
	Period          string
	RenewsOn        *time.Time
	Notes           string
	Assigned        int
}

// MonthlyMicros is the subscription's cost normalised to a month.
func (s Subscription) MonthlyMicros() int64 {
	total := int64(s.Seats) * s.SeatPriceMicros
	if s.Period == "year" {
		return total / 12
	}
	return total
}

// SubscriptionInput is what the subscription form sets.
type SubscriptionInput struct {
	Vendor          string
	Product         string
	Seats           int
	SeatPriceMicros int64
	Period          string
	RenewsOn        *time.Time
	Notes           string
}

func (in *SubscriptionInput) validate() error {
	if in.Period == "" {
		in.Period = "month"
	}
	for _, err := range []error{text("vendor", in.Vendor, 80, true), text("product", in.Product, 120, true), text("notes", in.Notes, 2000, false),
		oneOf("billing period", in.Period, "month", "year")} {
		if err != nil {
			return err
		}
	}
	if in.Seats < 0 || in.Seats > 1_000_000 || in.SeatPriceMicros < 0 {
		return Invalid("seats and prices cannot be negative")
	}
	return nil
}

// Subscriptions lists subscriptions with assigned seat counts.
func Subscriptions(ctx context.Context, q db.Querier) ([]Subscription, error) {
	rows, err := q.Query(ctx, `SELECT s.id, s.vendor, s.product, s.seats, s.seat_price_micros, s.billing_period, s.renews_on::timestamptz, s.notes,
		(SELECT count(*) FROM subscription_seats x WHERE x.subscription_id = s.id) FROM subscriptions s ORDER BY s.vendor, s.product`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Subscription
	for rows.Next() {
		var s Subscription
		if err := rows.Scan(&s.ID, &s.Vendor, &s.Product, &s.Seats, &s.SeatPriceMicros, &s.Period, &s.RenewsOn, &s.Notes, &s.Assigned); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

// SaveSubscription creates a subscription, or updates it when id is set.
func (t *Tx) SaveSubscription(ctx context.Context, id string, in SubscriptionInput) (string, error) {
	if err := in.validate(); err != nil {
		return "", err
	}
	var renews any
	if in.RenewsOn != nil {
		renews = in.RenewsOn.UTC().Format("2006-01-02")
	}
	action := "subscription.updated"
	if id == "" {
		id, action = ids.New(), "subscription.created"
		if _, err := t.tx.Exec(ctx, `INSERT INTO subscriptions (id, vendor, product, seats, seat_price_micros, billing_period, renews_on, notes)
			VALUES ($1,$2,$3,$4,$5,$6,$7::date,$8)`, id, in.Vendor, in.Product, in.Seats, in.SeatPriceMicros, in.Period, renews, in.Notes); err != nil {
			return "", err
		}
	} else {
		tag, err := t.tx.Exec(ctx, `UPDATE subscriptions SET vendor = $2, product = $3, seats = $4, seat_price_micros = $5, billing_period = $6,
			renews_on = $7::date, notes = $8, updated_at = now() WHERE id = $1`, id, in.Vendor, in.Product, in.Seats, in.SeatPriceMicros, in.Period, renews, in.Notes)
		if err != nil {
			return "", err
		}
		if tag.RowsAffected() == 0 {
			return "", ErrNotFound
		}
	}
	return id, t.record(ctx, action, "subscription", id, map[string]any{"vendor": in.Vendor, "product": in.Product, "seats": in.Seats})
}

// DeleteSubscription removes a subscription and its seat assignments.
func (t *Tx) DeleteSubscription(ctx context.Context, id string) error {
	tag, err := t.tx.Exec(ctx, "DELETE FROM subscriptions WHERE id = $1", id)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return t.record(ctx, "subscription.deleted", "subscription", id, nil)
}

// AssignSeat gives a user a seat, refusing when every seat is taken.
func (t *Tx) AssignSeat(ctx context.Context, subscriptionID, userID string) error {
	var seats, used int
	err := t.tx.QueryRow(ctx, `SELECT s.seats, (SELECT count(*) FROM subscription_seats x WHERE x.subscription_id = s.id)
		FROM subscriptions s WHERE s.id = $1 FOR UPDATE`, subscriptionID).Scan(&seats, &used)
	if err != nil {
		return notFound(err)
	}
	if used >= seats {
		return Invalid("every seat is assigned; add seats or reclaim one first")
	}
	tag, err := t.tx.Exec(ctx, "INSERT INTO subscription_seats (subscription_id, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", subscriptionID, userID)
	if err != nil || tag.RowsAffected() == 0 {
		return err
	}
	return t.record(ctx, "subscription.seat_assigned", "subscription", subscriptionID, map[string]any{"user_id": userID})
}

// UnassignSeat frees a user's seat.
func (t *Tx) UnassignSeat(ctx context.Context, subscriptionID, userID string) error {
	tag, err := t.tx.Exec(ctx, "DELETE FROM subscription_seats WHERE subscription_id = $1 AND user_id = $2", subscriptionID, userID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return t.record(ctx, "subscription.seat_unassigned", "subscription", subscriptionID, map[string]any{"user_id": userID})
}

// Seat is one assignment with the holder's state, for the reclamation report.
type Seat struct {
	SubscriptionID string
	Product        string
	UserID         string
	Email          string
	Status         string
	AssignedAt     time.Time
	LastUsed       *time.Time
}

// Reclaimable reports seats whose holder is not active or has not used Starlight in 30 days.
func (s Seat) Reclaimable() bool {
	return s.Status != "active" || s.LastUsed == nil || time.Since(*s.LastUsed) > 30*24*time.Hour
}

// Seats lists every assignment with the holder's status and last Starlight use.
func Seats(ctx context.Context, q db.Querier) ([]Seat, error) {
	rows, err := q.Query(ctx, `SELECT x.subscription_id, s.vendor || ' ' || s.product, u.id, u.email, u.status, x.assigned_at,
		(SELECT max(e.created_at) FROM usage_events e WHERE e.user_id = u.id AND e.created_at > now() - interval '90 days')
		FROM subscription_seats x JOIN subscriptions s ON s.id = x.subscription_id JOIN users u ON u.id = x.user_id
		ORDER BY s.vendor, s.product, u.email`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Seat
	for rows.Next() {
		var s Seat
		if err := rows.Scan(&s.SubscriptionID, &s.Product, &s.UserID, &s.Email, &s.Status, &s.AssignedAt, &s.LastUsed); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}
