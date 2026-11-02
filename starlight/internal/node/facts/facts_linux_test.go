package facts

import (
	"bufio"
	"strings"
	"testing"
)

func TestParseMeminfoAndOSRelease(t *testing.T) {
	sc := bufio.NewScanner(strings.NewReader("MemFree:  10 kB\nMemTotal:       263782348 kB\n"))
	if got := parseMeminfo(sc); got != 263782348<<10 {
		t.Fatalf("meminfo %d", got)
	}
	if got := osRelease("NAME=\"Ubuntu\"\nPRETTY_NAME=\"Ubuntu 24.04.3 LTS\"\n"); got != "Ubuntu 24.04.3 LTS" {
		t.Fatalf("os-release %q", got)
	}
}
