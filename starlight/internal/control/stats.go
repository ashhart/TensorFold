package control

import (
	"encoding/json"
	"math"
	"net/http"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
)

// stats records each replica's live load for dashboards; a node can only report its own replicas.
func (s *Server) stats(w http.ResponseWriter, r *http.Request) {
	nodeID, _, revoked, err := s.node(r)
	if err != nil || revoked {
		httpx.Write(w, httpx.Plain, errNode)
		return
	}
	var rep wire.StatsReport
	if err := decode(r, &rep); err != nil {
		httpx.Write(w, httpx.Plain, err)
		return
	}
	if len(rep.Replicas) > 256 {
		rep.Replicas = rep.Replicas[:256]
	}
	for _, st := range rep.Replicas {
		st.TokensPerSecond, st.PromptTokensPerSecond = finite(st.TokensPerSecond), finite(st.PromptTokensPerSecond)
		st.KVUsage = min(finite(st.KVUsage), 1)
		st.Running, st.Waiting = max(st.Running, 0), max(st.Waiting, 0)
		live, _ := json.Marshal(st)
		if _, err := s.Pool.Exec(r.Context(), "UPDATE replicas SET live = $3, live_at = now() WHERE id = $1 AND node_id = $2", st.ReplicaID, nodeID, live); err != nil {
			httpx.Write(w, httpx.Plain, err)
			return
		}
	}
	_, _ = s.Pool.Exec(r.Context(), "UPDATE nodes SET last_seen_at = now() WHERE id = $1", nodeID)
	w.WriteHeader(http.StatusNoContent)
}

func finite(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
		return 0
	}
	return v
}
