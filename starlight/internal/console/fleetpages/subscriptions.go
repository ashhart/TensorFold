package fleetpages

import (
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
)

type subscriptionsPage struct {
	format
	Subscriptions []fleet.Subscription
	Reclaim       []fleet.Seat
	Monthly       int64
	Seats         int
	Assigned      int
	CanWrite      bool
}

type subscriptionPage struct {
	format
	Subscription fleet.Subscription
	Seats        []fleet.Seat
	CanWrite     bool
}

func (p *pages) subscriptions() {
	p.c.Handle("GET /console/subscriptions", authz.SubscriptionsRead, p.subscriptionList)
	p.c.Handle("POST /console/subscriptions", authz.SubscriptionsWrite, p.subscriptionSave)
	p.c.Handle("GET /console/subscriptions/{id}", authz.SubscriptionsRead, p.subscriptionDetail)
	p.c.Handle("POST /console/subscriptions/{id}", authz.SubscriptionsWrite, p.subscriptionSave)
	p.c.Handle("POST /console/subscriptions/{id}/delete", authz.SubscriptionsWrite, p.subscriptionDelete)
	p.c.Handle("POST /console/subscriptions/{id}/seats", authz.SubscriptionsWrite, p.seatAssign)
	p.c.Handle("POST /console/subscriptions/{id}/seats/{user}/remove", authz.SubscriptionsWrite, p.seatRemove)
}

func subscriptionInput(q *console.Request) (fleet.SubscriptionInput, error) {
	seats, err := q.Int("seats", 0)
	if err != nil {
		return fleet.SubscriptionInput{}, err
	}
	price, err := fleet.ParseDollars("seat price", q.Form("seat_price"))
	if err != nil {
		return fleet.SubscriptionInput{}, friendly(err)
	}
	in := fleet.SubscriptionInput{Vendor: q.Form("vendor"), Product: q.Form("product"), Seats: seats, SeatPriceMicros: price,
		Period: q.Form("period"), Notes: q.Form("notes")}
	if v := q.Form("renews_on"); v != "" {
		t, err := time.Parse(day, v)
		if err != nil {
			return in, console.Invalid("renewal must be a date")
		}
		in.RenewsOn = &t
	}
	return in, nil
}

// subscription finds one subscription by its path id.
func (p *pages) subscription(q *console.Request) (fleet.Subscription, error) {
	list, err := fleet.Subscriptions(q.Ctx(), p.c.Pool)
	if err != nil {
		return fleet.Subscription{}, err
	}
	for _, s := range list {
		if s.ID == q.R.PathValue("id") {
			return s, nil
		}
	}
	return fleet.Subscription{}, friendly(fleet.ErrNotFound)
}

func (p *pages) subscriptionList(q *console.Request) error {
	list, err := fleet.Subscriptions(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	seats, err := fleet.Seats(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	page := subscriptionsPage{Subscriptions: list, CanWrite: q.Can(authz.SubscriptionsWrite, "")}
	for _, s := range list {
		page.Monthly += s.MonthlyMicros()
		page.Seats += s.Seats
		page.Assigned += s.Assigned
	}
	for _, s := range seats {
		if s.Reclaimable() {
			page.Reclaim = append(page.Reclaim, s)
		}
	}
	q.Render("fleet/subscriptions", "Subscriptions", page)
	return nil
}

func (p *pages) subscriptionDetail(q *console.Request) error {
	s, err := p.subscription(q)
	if err != nil {
		return err
	}
	seats, err := fleet.Seats(q.Ctx(), p.c.Pool)
	if err != nil {
		return err
	}
	page := subscriptionPage{Subscription: s, CanWrite: q.Can(authz.SubscriptionsWrite, "")}
	for _, seat := range seats {
		if seat.SubscriptionID == s.ID {
			page.Seats = append(page.Seats, seat)
		}
	}
	q.Render("fleet/subscription", s.Vendor+" "+s.Product, page)
	return nil
}

func (p *pages) subscriptionSave(q *console.Request) error {
	id := q.R.PathValue("id")
	if id != "" && !validID(id) {
		return friendly(fleet.ErrNotFound)
	}
	in, err := subscriptionInput(q)
	if err != nil {
		return err
	}
	if err := p.do(q, func(tx *fleet.Tx) (err error) { id, err = tx.SaveSubscription(q.Ctx(), id, in); return err }); err != nil {
		return err
	}
	return q.Redirect("/console/subscriptions/"+id, "Subscription saved.")
}

func (p *pages) subscriptionDelete(q *console.Request) error {
	s, err := p.subscription(q)
	if err != nil {
		return err
	}
	if !q.Bool("confirm") {
		return console.Invalid("Tick the confirmation box to delete " + s.Vendor + " " + s.Product + ".")
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.DeleteSubscription(q.Ctx(), s.ID) }); err != nil {
		return err
	}
	return q.Redirect("/console/subscriptions", "Subscription deleted.")
}

func (p *pages) seatAssign(q *console.Request) error {
	s, err := p.subscription(q)
	if err != nil {
		return err
	}
	u, err := fleet.UserByEmail(q.Ctx(), p.c.Pool, q.Form("email"))
	if err != nil {
		return friendly(err)
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.AssignSeat(q.Ctx(), s.ID, u.ID) }); err != nil {
		return err
	}
	return q.Redirect("/console/subscriptions/"+s.ID, "Seat assigned to "+u.Email+".")
}

func (p *pages) seatRemove(q *console.Request) error {
	s, err := p.subscription(q)
	if err != nil {
		return err
	}
	user := q.R.PathValue("user")
	if !validID(user) {
		return friendly(fleet.ErrNotFound)
	}
	if err := p.do(q, func(tx *fleet.Tx) error { return tx.UnassignSeat(q.Ctx(), s.ID, user) }); err != nil {
		return err
	}
	back := "/console/subscriptions/" + s.ID
	if q.Form("from") == "report" {
		back = "/console/subscriptions#reclaim"
	}
	return q.Redirect(back, "Seat freed.")
}
