package engine

import "testing"

func TestEngineReleaseGate(t *testing.T) {
	for line, old := range map[string]bool{"tensorfold-native 1.1.6": true, "tensorfold 1.1.9": true, "tensorfold-native 1.2.0": false,
		"tensorfold 1.2.0-fake": false, "tensorfold-native 9.8.7": false, "tensorfold 2.0.1": false} {
		v, ok := parseVersion(line)
		if !ok || older(v, MinEngine) != old {
			t.Errorf("%q: older than %v = %v", line, MinEngine, older(v, MinEngine))
		}
	}
	if _, ok := parseVersion("tensorfold dev"); ok {
		t.Error("a version line without x.y.z parsed")
	}
}
