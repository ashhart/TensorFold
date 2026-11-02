//go:build windows

package managed

import (
	"errors"
	"strings"

	"golang.org/x/sys/windows/registry"
)

// PolicyKey is the group policy registry path, read under HKCU and then HKLM so machine policy wins.
const PolicyKey = `SOFTWARE\Policies\TensorFold\Connect`

func loadPlatform() (Policy, error) {
	var out Policy
	for _, root := range []struct {
		key  registry.Key
		name string
	}{{registry.CURRENT_USER, `HKCU\` + PolicyKey}, {registry.LOCAL_MACHINE, `HKLM\` + PolicyKey}} {
		k, err := registry.OpenKey(root.key, PolicyKey, registry.QUERY_VALUE)
		if errors.Is(err, registry.ErrNotExist) {
			continue
		}
		if err != nil {
			return Policy{}, err
		}
		p := readKey(k)
		k.Close()
		if !p.Empty() {
			p.Sources = []string{root.name}
		}
		out = Merge(out, p)
	}
	return out, nil
}

func readKey(k registry.Key) Policy {
	str := func(name string) string {
		if s, _, err := k.GetStringValue(name); err == nil {
			return s
		}
		if lines, _, err := k.GetStringsValue(name); err == nil {
			return strings.Join(lines, "\n")
		}
		return ""
	}
	num := func(name string) (uint64, bool) {
		v, _, err := k.GetIntegerValue(name)
		return v, err == nil
	}
	p := Policy{ServerURL: str("ServerURL"), CACertificatePEM: str("CACertificatePEM"), CAFile: str("CAFile")}
	if v, ok := num("AllowManualServer"); ok {
		allow := v != 0
		p.AllowManualServer = &allow
	}
	if v, ok := num("BridgePort"); ok && v <= 65535 {
		p.BridgePort = int(v)
	}
	if v, ok := num("DisableBridge"); ok {
		p.DisableBridge = v != 0
	}
	return p
}
