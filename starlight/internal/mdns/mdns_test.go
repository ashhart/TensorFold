package mdns

import (
	"context"
	"net"
	"testing"
	"time"

	hmdns "github.com/hashicorp/mdns"
)

func TestTXT(t *testing.T) {
	txt, err := TXT(Config{URL: "https://starlight.example", InstallationID: "0190F5A0-1234-7000-8000-000000000001"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"v=1", "url=https://starlight.example", "id=0190f5a0"}
	for i := range want {
		if txt[i] != want[i] {
			t.Fatalf("txt = %v", txt)
		}
	}
	for _, bad := range []Config{{URL: "starlight.example", InstallationID: "abc"}, {URL: "https://starlight.example", InstallationID: "!!"},
		{URL: "https://u:p@starlight.example", InstallationID: "abc"}} {
		if _, err := TXT(bad); err == nil {
			t.Errorf("accepted %+v", bad)
		}
	}
}

func TestParseRequiresVersionAndURL(t *testing.T) {
	ok := &hmdns.ServiceEntry{Name: "starlight-a." + Service + ".local.", Host: "a.local.", Port: 443, AddrV4: net.ParseIP("192.0.2.10"),
		InfoFields: []string{"v=1", "url=https://starlight.example", "id=0190f5a0"}}
	f, good := parse(ok)
	if !good || f.Instance != "starlight-a" || f.URL != "https://starlight.example" || f.InstallationID != "0190f5a0" || len(f.Addrs) != 1 {
		t.Fatalf("parse = %+v, %v", f, good)
	}
	for _, fields := range [][]string{{"v=2", "url=https://starlight.example"}, {"v=1", "url=javascript:x"}, {"v=1"}} {
		if _, good := parse(&hmdns.ServiceEntry{InfoFields: fields}); good {
			t.Errorf("accepted %v", fields)
		}
	}
}

func TestNames(t *testing.T) {
	if got := instanceName("TensorFold Starlight (HQ)"); got != "tensorfold-starlight-hq" {
		t.Fatalf("instance %q", got)
	}
	if got := instanceName("!!!"); got != "tensorfold-starlight" {
		t.Fatalf("empty instance %q", got)
	}
	if h := hostName(); len(h) < 7 || h[len(h)-7:] != ".local." {
		t.Fatalf("host %q", h)
	}
}

func TestAdvertiseAndBrowse(t *testing.T) {
	if testing.Short() {
		t.Skip("multicast test skipped in short mode")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cfg := Config{Instance: "connect-test-" + time.Now().Format("150405"), Port: 8443, URL: "https://starlight-test.example", InstallationID: "abcdef12"}
	stop, err := Advertise(ctx, cfg)
	if err != nil {
		t.Skipf("multicast unavailable: %v", err)
	}
	defer stop()
	found, err := Browse(ctx, 2*time.Second)
	if err != nil {
		t.Skipf("multicast browse unavailable: %v", err)
	}
	for _, f := range found {
		if f.URL == cfg.URL && f.InstallationID == "abcdef12" && f.Port == 8443 {
			return
		}
	}
	if len(found) == 0 {
		t.Skip("no mDNS answers on this network; multicast is likely filtered")
	}
	t.Fatalf("own advertisement not found among %+v", found)
}

func TestCollectReadsEntriesOnlyAfterTheQueryReturns(t *testing.T) {
	got, err := collect(func(entries chan<- *hmdns.ServiceEntry) error {
		e := &hmdns.ServiceEntry{Name: "a." + Service + ".local.", Host: "a.local.", Port: 8443,
			InfoFields: []string{"v=1", "url=https://starlight.example", "id=abc"}}
		entries <- e
		for i := range 200 {
			e.InfoFields = []string{"v=1", "url=https://starlight.example", "id=abc"}
			e.AddrV4 = net.IPv4(192, 0, 2, byte(i))
			time.Sleep(50 * time.Microsecond)
		}
		return nil
	})
	if err != nil || len(got) != 1 || got[0].URL != "https://starlight.example" || got[0].Instance != "a" || len(got[0].Addrs) != 1 {
		t.Fatalf("collect = %+v, %v", got, err)
	}
}
