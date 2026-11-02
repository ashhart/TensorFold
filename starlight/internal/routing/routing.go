// Package routing chooses among eligible targets: affinity for prompt-cache reuse, otherwise the most headroom.
package routing

import (
	"encoding/binary"
	"hash/fnv"
	"math/rand/v2"
)

// Candidate is one target with spare capacity.
type Candidate struct {
	ID    string
	Free  int
	Slots int
}

// affinityFloor keeps cache affinity from piling work onto a nearly full replica.
const affinityFloor = 0.25

// Pick returns the index of the chosen candidate, or -1 when none has a free slot.
func Pick(cands []Candidate, affinity uint64) int {
	best, bestScore := -1, -1.0
	if affinity != 0 {
		var top uint64
		for i, c := range cands {
			if c.Free <= 0 || c.Slots <= 0 || float64(c.Free)/float64(c.Slots) < affinityFloor {
				continue
			}
			if h := weight(c.ID, affinity); best < 0 || h > top {
				best, top = i, h
			}
		}
		if best >= 0 {
			return best
		}
	}
	for i, c := range cands {
		if c.Free <= 0 || c.Slots <= 0 {
			continue
		}
		score := float64(c.Free)/float64(c.Slots) + rand.Float64()*1e-6
		if score > bestScore {
			best, bestScore = i, score
		}
	}
	return best
}

// weight is rendezvous hashing, so a prompt prefix keeps its replica while the set of replicas changes.
func weight(id string, key uint64) uint64 {
	h := fnv.New64a()
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], key)
	h.Write(b[:])
	h.Write([]byte(id))
	return h.Sum64()
}

// Affinity hashes the first part of a prompt; equal prefixes land on the same replica.
func Affinity(prefix []byte) uint64 {
	if len(prefix) == 0 {
		return 0
	}
	if len(prefix) > 4096 {
		prefix = prefix[:4096]
	}
	h := fnv.New64a()
	h.Write(prefix)
	if s := h.Sum64(); s != 0 {
		return s
	}
	return 1
}
