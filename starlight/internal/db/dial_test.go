package db

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"syscall"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestDialerHoldsASlotUntilTheServerAnswers(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			defer c.Close()
		}
	}()
	d := &dialer{slots: make(chan struct{}, 2)}
	ctx := context.Background()
	a, err := d.dial(ctx, "tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.dial(ctx, "tcp", ln.Addr().String()); err != nil {
		t.Fatal(err)
	}
	third := make(chan error, 1)
	go func() {
		_, err := d.dial(ctx, "tcp", ln.Addr().String())
		third <- err
	}()
	select {
	case <-third:
		t.Fatal("a third connection opened while two still waited for the server")
	case <-time.After(150 * time.Millisecond):
	}
	a.Close()
	select {
	case err := <-third:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("closing a waiting connection did not free its slot")
	}
}

func TestUnreachableIsUnavailable(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := ln.Addr().String()
	ln.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := Open(ctx, "postgres://starlight@"+addr+"/postgres?sslmode=disable&connect_timeout=2", Options{})
	if err == nil || !Unavailable(err) {
		t.Fatalf("open on a closed port = %v, unavailable %v", err, Unavailable(err))
	}
}

func TestUnavailableCoversOutages(t *testing.T) {
	for _, err := range []error{context.DeadlineExceeded, &pgconn.PgError{Code: "08006"}, &pgconn.PgError{Code: "57P01"},
		&pgconn.PgError{Code: "57P02"}, &pgconn.PgError{Code: "57P03"}, &pgconn.PgError{Code: "57014"}, &pgconn.PgError{Code: "53300"},
		&pgconn.PgError{Code: "40P01"}, io.ErrUnexpectedEOF, syscall.ECONNRESET, pgconn.ErrConnClosed,
		&url.Error{Op: "Get", URL: "https://idp.example.com/keys", Err: &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}}} {
		if !Unavailable(fmt.Errorf("query: %w", err)) {
			t.Errorf("%v is not reported unavailable", err)
		}
	}
	for _, err := range []error{nil, context.Canceled, errors.New("boom"), &pgconn.PgError{Code: "23505"}, &pgconn.PgError{Code: "42P01"}} {
		if Unavailable(err) {
			t.Errorf("%v is reported unavailable", err)
		}
	}
}

func TestConnectingIsBoundedByDefault(t *testing.T) {
	cfg, err := poolConfig("postgres://starlight@db.example.com/starlight", Options{})
	if err != nil || cfg.ConnConfig.ConnectTimeout != 10*time.Second {
		t.Fatalf("connect timeout without connect_timeout: %v %v", cfg.ConnConfig.ConnectTimeout, err)
	}
	if cfg, err = poolConfig("postgres://starlight@db.example.com/starlight?connect_timeout=3", Options{}); err != nil || cfg.ConnConfig.ConnectTimeout != 3*time.Second {
		t.Fatalf("connect_timeout=3 gave %v %v", cfg.ConnConfig.ConnectTimeout, err)
	}
}
