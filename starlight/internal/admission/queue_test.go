package admission_test

import (
	"context"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

func TestRefusedWakeUpIsHandedOn(t *testing.T) {
	f := newFleet(t, 1)
	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	w := admission.NewWaker()
	go w.Run(ctx, f.pool)
	step := 300 * time.Millisecond
	a := &admission.Admitter{Pool: f.pool, GatewayID: ids.New(), Poll: time.Minute, Ageing: step, Waker: w}
	holder, err := a.Admit(ctx, f.request(levels.Standard, time.Second))
	if err != nil {
		t.Fatal(err)
	}
	admitted := make(chan time.Time, 1)
	go func() {
		if l, err := a.Admit(ctx, f.request(levels.Batch, time.Minute)); err == nil {
			admitted <- time.Now()
			_ = a.Settle(ctx, l, admission.Usage{Observed: true}, record())
		}
	}()
	// Aged a full class, the batch waiter ties a fresh standard one and goes first as the older, but is woken second.
	time.Sleep(11 * step)
	standard := f.request(levels.Standard, time.Minute)
	go func() {
		if l, err := a.Admit(ctx, standard); err == nil {
			_ = a.Settle(ctx, l, admission.Usage{Observed: true}, record())
		}
	}()
	eventually(t, "the standard request queues", func() bool {
		var queued int
		_ = f.pool.QueryRow(ctx, "SELECT count(*) FROM admission_queue WHERE request_id = $1", standard.RequestID).Scan(&queued)
		return queued > 0
	})
	freed := time.Now()
	if err := a.Settle(ctx, holder, admission.Usage{Observed: true}, record()); err != nil {
		t.Fatal(err)
	}
	select {
	case at := <-admitted:
		if at.Sub(freed) > 2*time.Second {
			t.Fatalf("the aged batch waiter took %v to get the freed slot", at.Sub(freed))
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the standard waiter's refused wake-up never reached the aged batch waiter that outranks it")
	}
}
