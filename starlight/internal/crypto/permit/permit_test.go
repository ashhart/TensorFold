package permit

import (
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"strings"
	"testing"
	"time"
)

func fixture(t *testing.T) (Signer, *Verifier, Claims) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	c := Claims{RequestID: "r1", Node: "n1", Replica: "rep1", Deployment: "d1", Revision: 3, Method: "POST", Path: "/v1/chat/completions",
		BodySHA256: strings.Repeat("a", 64), IssuedAt: now.UnixMilli(), Expires: now.Add(30 * time.Second).UnixMilli(), Subject: "s", Priority: "standard"}
	return Signer{KID: "k1", Key: priv}, NewVerifier(map[string]ed25519.PublicKey{"k1": pub}), c
}

func TestRoundTrip(t *testing.T) {
	s, v, c := fixture(t)
	tok, err := s.Sign(c)
	if err != nil {
		t.Fatal(err)
	}
	got, err := v.Verify(tok, time.Now())
	if err != nil || got != c {
		t.Fatalf("verify = %+v, %v", got, err)
	}
}

func TestRejects(t *testing.T) {
	s, v, c := fixture(t)
	tok, _ := s.Sign(c)
	parts := strings.Split(tok, ".")
	other, _, _ := fixture(t)
	foreign, _ := other.Sign(c)
	cases := map[string]error{
		"v2." + strings.Join(parts[1:], "."):               ErrMalformed,
		parts[0] + ".k9." + parts[2] + "." + parts[3]:      ErrKey,
		parts[0] + "." + parts[1] + "." + parts[2] + ".AA": ErrMalformed,
		foreign: ErrSignature,
	}
	for in, want := range cases {
		if _, err := v.Verify(in, time.Now()); !errors.Is(err, want) {
			t.Errorf("verify(%.20s) = %v, want %v", in, err, want)
		}
	}
	if _, err := v.Verify(tok, time.Now().Add(time.Minute)); !errors.Is(err, ErrExpired) {
		t.Errorf("late verify = %v", err)
	}
	long := c
	long.Expires = time.UnixMilli(c.IssuedAt).Add(10 * time.Minute).UnixMilli()
	tl, _ := s.Sign(long)
	if _, err := v.Verify(tl, time.Now()); !errors.Is(err, ErrMalformed) {
		t.Errorf("over-long permit = %v", err)
	}
}

func TestReplay(t *testing.T) {
	r := NewReplay()
	now := time.Now()
	if !r.Use("a", now.Add(time.Second), now) || r.Use("a", now.Add(time.Second), now) {
		t.Fatal("replay cache did not block a second use")
	}
	if !r.Use("b", now.Add(time.Second), now) {
		t.Fatal("blocked a fresh id")
	}
}
