package fakeengine

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"
)

// rankPrefixes are the families whose rank variables the fake engine reads, as the real engine does.
var rankPrefixes = []string{"TF_GLM", "TF_NEMOTRON", "TF_MINIMAX", "TF_TP"}

// place returns this process's rank, world and rank 0's address; world 1 means it runs alone.
func place() (rank, world int, master string) {
	for _, p := range rankPrefixes {
		if w, err := strconv.Atoi(os.Getenv(p + "_WORLD")); err == nil && w > 1 {
			rank, _ = strconv.Atoi(os.Getenv(p + "_RANK"))
			world = w
			break
		}
	}
	master = os.Getenv("NCCL_COMM_ID")
	if m := os.Getenv("TF_TP_MASTER"); m != "" {
		master = net.JoinHostPort(m, os.Getenv("TF_TP_PORT"))
	}
	return rank, max(world, 1), master
}

// followerRank reports a tensor-parallel rank past 0, which follows rank 0 instead of serving.
func followerRank() (string, bool) {
	rank, world, _ := place()
	return strconv.Itoa(rank), world > 1 && rank > 0
}

// recordRank writes this process's placement into FAKE_ENGINE_RANK_DIR so tests can check GPU pinning.
func recordRank(rank string) {
	dir := knob("FAKE_ENGINE_RANK_DIR")
	if dir == "" {
		return
	}
	_, world, master := place()
	b, _ := json.Marshal(map[string]string{"rank": rank, "world": strconv.Itoa(world), "visible": os.Getenv("CUDA_VISIBLE_DEVICES"),
		"order": os.Getenv("CUDA_DEVICE_ORDER"), "comm": master, "pid": fmt.Sprint(os.Getpid())})
	_ = os.WriteFile(filepath.Join(dir, "rank"+rank+".json"), b, 0o600)
}

// gather stands in for NCCL's bootstrap on rank 0: every rank must connect before it serves, and any drop ends it.
func gather(stderr io.Writer) error {
	_, world, master := place()
	if world <= 1 || master == "" {
		return nil
	}
	ln, err := net.Listen("tcp", master)
	if err != nil {
		return fmt.Errorf("rank 0 cannot listen at %s: %w", master, err)
	}
	defer ln.Close()
	_ = ln.(*net.TCPListener).SetDeadline(time.Now().Add(rendezvous()))
	for joined := 1; joined < world; joined++ {
		c, err := ln.Accept()
		if err != nil {
			return fmt.Errorf("only %d of %d ranks joined: %w", joined, world, err)
		}
		go func() {
			_, _ = io.Copy(io.Discard, c)
			fmt.Fprintln(stderr, "fatal: a rank left the group")
			os.Exit(5)
		}()
	}
	return nil
}

// follow joins rank 0 and idles until stopped; it exits when rank 0 goes away, or on FAKE_ENGINE_FOLLOWER_CRASH_MS.
func follow(rank string, stderr io.Writer) int {
	recordRank(rank)
	_, _, master := place()
	lost := make(chan struct{})
	if master != "" {
		deadline := time.Now().Add(rendezvous())
		var c net.Conn
		var err error
		for c, err = net.Dial("tcp", master); err != nil && time.Now().Before(deadline); c, err = net.Dial("tcp", master) {
			time.Sleep(50 * time.Millisecond)
		}
		if err != nil {
			fmt.Fprintln(stderr, "fatal: rank 0 never answered at", master)
			return 4
		}
		go func() {
			_, _ = bufio.NewReader(c).ReadByte()
			close(lost)
		}()
	}
	fmt.Fprintf(stderr, "rank %s following rank 0\n", rank)
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, os.Interrupt)
	var crash <-chan time.Time
	if ms := envMillis("FAKE_ENGINE_FOLLOWER_CRASH_MS"); ms > 0 {
		crash = time.After(ms)
	}
	select {
	case <-sig:
		return 0
	case <-lost:
		fmt.Fprintln(stderr, "fatal: lost rank 0")
		return 4
	case <-crash:
		fmt.Fprintln(stderr, "fatal: lost rank 0")
		return 4
	}
}

func rendezvous() time.Duration {
	if ms := envMillis("FAKE_ENGINE_RENDEZVOUS_MS"); ms > 0 {
		return ms
	}
	return 60 * time.Second
}
