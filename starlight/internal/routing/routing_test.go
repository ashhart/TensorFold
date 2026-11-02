package routing

import (
	"fmt"
	"testing"
)

func TestPickPrefersHeadroomWithoutAffinity(t *testing.T) {
	c := []Candidate{{"a", 1, 8}, {"b", 6, 8}, {"c", 0, 8}}
	if got := Pick(c, 0); got != 1 {
		t.Fatalf("picked %d", got)
	}
	if Pick([]Candidate{{"a", 0, 4}}, 0) != -1 {
		t.Fatal("picked a full candidate")
	}
}

func TestAffinityIsStableAndRespectsFloor(t *testing.T) {
	c := []Candidate{{"a", 4, 8}, {"b", 4, 8}, {"c", 4, 8}}
	key := Affinity([]byte("You are a coding agent. Repository: starlight."))
	first := Pick(c, key)
	for i := 0; i < 20; i++ {
		if Pick(c, key) != first {
			t.Fatal("affinity not stable")
		}
	}
	c[first].Free = 1
	if Pick(c, key) == first {
		t.Fatal("affinity ignored the headroom floor")
	}
	spread := map[int]bool{}
	for i := 0; i < 200; i++ {
		spread[Pick([]Candidate{{"a", 4, 8}, {"b", 4, 8}, {"c", 4, 8}}, Affinity([]byte(fmt.Sprint("prompt", i))))] = true
	}
	if len(spread) != 3 {
		t.Fatalf("affinity did not spread prompts: %v", spread)
	}
}
