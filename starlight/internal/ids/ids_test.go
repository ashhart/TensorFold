package ids

import "testing"

func TestNewIsOrderedAndValid(t *testing.T) {
	prev := ""
	for i := 0; i < 10000; i++ {
		id := New()
		if !Valid(id) {
			t.Fatalf("invalid id %q", id)
		}
		if id[14] != '7' {
			t.Fatalf("not version 7: %q", id)
		}
		if id <= prev {
			t.Fatalf("ids not increasing: %q then %q", prev, id)
		}
		prev = id
	}
}

func TestValidRejects(t *testing.T) {
	for _, s := range []string{"", "x", "0190f5a0-0000-7000-8000-00000000000G", "0190F5A0-0000-7000-8000-000000000000", "0190f5a000000-7000-8000-000000000000"} {
		if Valid(s) {
			t.Errorf("accepted %q", s)
		}
	}
}
