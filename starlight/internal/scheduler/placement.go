package scheduler

import (
	"maps"
	"sort"
	"strconv"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// seat is one node's part of a replica being placed and the GPUs it will pin there.
type seat struct {
	node    *node
	devices []string
	score   float64
}

// Groups meet on blocks of wire.RankPorts ports from rankBase on their lead node.
const (
	rankBase   = 29500
	rankBlocks = 100
)

// place picks the nodes for one replica, its lead first: spread, then a cached artifact, then dedicated GPUs, then fit.
func (s *snapshot) place(d deployment) []seat {
	hosting := s.hosting(d.id)
	span := d.spec.Span()
	share := divUp(uint64(d.spec.MemoryGiB)<<30, uint64(span))
	var offers []seat
	for _, n := range s.nodes {
		if o, ok := s.offer(n, d, share, hosting[n.id]); ok {
			offers = append(offers, o)
		}
	}
	if len(offers) < span {
		return nil
	}
	sort.SliceStable(offers, func(i, j int) bool { return offers[i].score > offers[j].score })
	chosen := offers[:span]
	for _, c := range chosen {
		if c.node.dedicated() {
			for _, id := range c.devices {
				c.node.taken[id] = true
			}
		} else {
			c.node.reserved += share
		}
	}
	return chosen
}

// hosting lists the nodes where a replica of dep runs or is still draining, so a new one never lands on a stopping one.
func (s *snapshot) hosting(dep string) map[string]bool {
	out := map[string]bool{}
	for _, r := range s.replicas[dep] {
		for _, h := range r.seats() {
			if r.live() || r.draining() && r.holds(h) {
				out[h.node] = true
			}
		}
	}
	return out
}

// offer is what node n can give one seat of d: whole GPUs on dedicated memory, or memory on a unified node.
func (s *snapshot) offer(n *node, d deployment, share uint64, hosting bool) (seat, bool) {
	if n.pool != d.pool || s.failedOn[d.id][n.id] || !matches(n.labels, d.selector) || d.spec.Backend != "" && n.backend != d.spec.Backend {
		return seat{}, false
	}
	if sc, ok := wire.RankSchemes[d.spec.TensorParallel]; ok && n.backend != sc.Backend {
		return seat{}, false
	}
	o := seat{node: n}
	if n.dedicated() {
		if o.devices = n.free(d.spec.Width(), share); o.devices == nil {
			return seat{}, false
		}
		o.score = 1e5 - float64(n.freeGPUs()-len(o.devices))
		if hosting {
			o.score -= 1e7
		}
	} else {
		free := float64(n.memory) - float64(n.reserved)
		// a node whose replicas still pin GPUs it no longer reports has lost a GPU probe, not become unified memory
		if hosting || len(n.taken) > 0 || d.spec.Width() > 1 || share > 0 && (n.memory == 0 || free < float64(share)) {
			return seat{}, false
		}
		o.score = free / float64(1<<30)
	}
	if n.cached[d.spec.Model.Key()] {
		o.score += 1e6
	}
	return o, true
}

// makesRoom reports whether stopping r would let a replica of d be placed, so a rollout never stops in vain.
func (s *snapshot) makesRoom(d deployment, r replica) bool {
	return s.without(r).place(d) != nil
}

// without is a copy of the snapshot in which r has released everything it holds.
func (s *snapshot) without(r replica) *snapshot {
	c := &snapshot{deployments: s.deployments, failedOn: s.failedOn, queue: s.queue, replicas: map[string][]replica{}}
	byID := map[string]*node{}
	for _, n := range s.nodes {
		cp := *n
		cp.taken, cp.ports = maps.Clone(n.taken), maps.Clone(n.ports)
		c.nodes = append(c.nodes, &cp)
		byID[cp.id] = &cp
	}
	for dep, rs := range s.replicas {
		for _, x := range rs {
			if x.id != r.id {
				c.replicas[dep] = append(c.replicas[dep], x)
			}
		}
	}
	for i, h := range r.seats() {
		if n := byID[h.node]; n != nil && r.holds(h) {
			for _, id := range h.devices {
				delete(n.taken, id)
			}
			n.reserved -= min(n.reserved, r.share)
			if i == 0 {
				delete(n.ports, r.port)
			}
		}
	}
	return c
}

// freePort takes the lowest rendezvous block free on n, or 0 when every block is in use.
func (n *node) freePort() int {
	if n.ports == nil {
		n.ports = map[int]bool{}
	}
	for i := 0; i < rankBlocks; i++ {
		if p := rankBase + i*wire.RankPorts; !n.ports[p] {
			n.ports[p] = true
			return p
		}
	}
	return 0
}

// dedicated reports a node whose GPUs have memory of their own, so each replica holds whole GPUs there.
func (n *node) dedicated() bool {
	if n.backend != "cuda" && n.backend != "hip" {
		return false
	}
	for _, g := range n.gpus {
		if g.MemoryBytes > 0 {
			return true
		}
	}
	return false
}

// free returns the first width untaken GPUs with memory of their own that can each hold an even share of need, or nil.
func (n *node) free(width int, need uint64) []string {
	share := divUp(need, uint64(width))
	var out []string
	for _, g := range n.gpus {
		if id := DeviceID(g); g.MemoryBytes > 0 && !n.taken[id] && g.MemoryBytes >= share {
			if out = append(out, id); len(out) == width {
				return out
			}
		}
	}
	return nil
}

func (n *node) freeGPUs() int {
	free := 0
	for _, g := range n.gpus {
		if g.MemoryBytes > 0 && !n.taken[DeviceID(g)] {
			free++
		}
	}
	return free
}

// DeviceID names a GPU as CUDA_VISIBLE_DEVICES takes it: by UUID when the node reports one, else by physical index.
func DeviceID(g wire.GPU) string {
	if g.UUID != "" {
		return g.UUID
	}
	return strconv.Itoa(g.Index)
}

func matches(labels, selector map[string]string) bool {
	for k, v := range selector {
		if labels[k] != v {
			return false
		}
	}
	return true
}
