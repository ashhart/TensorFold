package wire

import (
	"errors"
	"fmt"
	"maps"
	"net"
	"regexp"
	"slices"
	"strings"
)

// RankScheme is how an engine family's tensor-parallel ranks find rank 0, each rank one process on one GPU.
type RankScheme struct {
	Prefix  string
	Meet    string
	Backend string
}

// Rendezvous kinds: NCCL's root address, the engine's own fabric (TF_TP_MASTER, TF_TP_PORT), or serve flags.
const (
	MeetNCCL   = "nccl"
	MeetFabric = "fabric"
	MeetFlags  = "flags"
)

// RankSchemes are the engine families whose ranks Starlight places, on one node or across several.
var RankSchemes = map[string]RankScheme{
	"glm":      {Prefix: "TF_GLM", Meet: MeetNCCL, Backend: "cuda"},
	"nemotron": {Prefix: "TF_NEMOTRON", Meet: MeetNCCL, Backend: "cuda"},
	"minimax":  {Prefix: "TF_MINIMAX", Meet: MeetNCCL, Backend: "cuda"},
	"deepseek": {Prefix: "TF_TP", Meet: MeetFabric, Backend: "cuda"},
	"rocm":     {Meet: MeetFlags, Backend: "hip"},
}

// RankPorts is how many consecutive ports, from its base, one group's rendezvous may use on rank 0's node.
const RankPorts = 10

// Group places a replica's ranks across nodes: this member runs ranks FirstRank to FirstRank+Width-1 of World.
type Group struct {
	Member    int    `json:"member"`
	World     int    `json:"world"`
	FirstRank int    `json:"first_rank"`
	Master    string `json:"master"`
	Port      int    `json:"port"`
}

var hostPattern = regexp.MustCompile(`^[A-Za-z0-9]([A-Za-z0-9-]{0,62})(\.[A-Za-z0-9]([A-Za-z0-9-]{0,62}))*$`)

// Width is how many GPUs, and so ranks, one replica holds on each node with dedicated GPU memory.
func (s Spec) Width() int { return max(s.GPUs, 1) }

// Span is how many nodes one replica runs on.
func (s Spec) Span() int { return max(s.Nodes, 1) }

// Ranks is the replica's tensor-parallel world: its GPUs on each node times its nodes.
func (s Spec) Ranks() int { return s.Width() * s.Span() }

func (s Spec) validateRanks() error {
	if s.TensorParallel == "" {
		if s.Ranks() > 1 {
			return errors.New("gpus or nodes above 1 split one model over ranks; set tensor_parallel to " + schemeNames())
		}
		return nil
	}
	sc, ok := RankSchemes[s.TensorParallel]
	switch {
	case !ok:
		return fmt.Errorf("unknown tensor_parallel %q; Starlight places %s ranks", s.TensorParallel, schemeNames())
	case s.Ranks() < 2:
		return fmt.Errorf("tensor_parallel %s needs two or more ranks: gpus or nodes above 1", s.TensorParallel)
	case s.Backend != "" && s.Backend != sc.Backend:
		return fmt.Errorf("tensor_parallel %s runs on %s", s.TensorParallel, sc.Backend)
	case sc.Meet == MeetFlags && !slices.Contains([]int{2, 4, 8}, s.Ranks()):
		return fmt.Errorf("tensor_parallel %s runs 2, 4 or 8 ranks", s.TensorParallel)
	}
	return nil
}

func schemeNames() string {
	return strings.Join(slices.Sorted(maps.Keys(RankSchemes)), ", ")
}

// Validate checks a member's placement against the spec it serves.
func (g Group) Validate(s Spec) error {
	if s.TensorParallel == "" || s.Span() < 2 || g.World != s.Ranks() || g.Member < 0 || g.Member >= s.Span() || g.FirstRank != g.Member*s.Width() {
		return errors.New("the rank group does not match its deployment spec")
	}
	if g.Port < 1024 || g.Port > 65535-RankPorts {
		return fmt.Errorf("rank port %d is outside 1024 to %d", g.Port, 65535-RankPorts)
	}
	if net.ParseIP(g.Master) == nil && !hostPattern.MatchString(g.Master) {
		return fmt.Errorf("rank 0's address %q is not a host", g.Master)
	}
	return nil
}
