package engine

import (
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// reap stops engines a previous agent left running, identified by PID and their unique key-file argument.
func reap(root string, logger *slog.Logger) {
	dirs, err := os.ReadDir(root)
	if err != nil {
		return
	}
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		files, _ := filepath.Glob(filepath.Join(root, d.Name(), "engine*.pid"))
		for _, f := range files {
			reapOne(f, d.Name(), logger)
		}
	}
}

// reapOne stops the process one pid file names, if its command line still carries the replica's key file.
func reapOne(pidFile, replica string, logger *slog.Logger) {
	b, err := os.ReadFile(pidFile)
	if err != nil {
		return
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	pid, err := strconv.Atoi(strings.TrimSpace(lines[0]))
	if err != nil || pid <= 1 || len(lines) < 2 {
		_ = os.Remove(pidFile)
		return
	}
	keyFile := strings.TrimSpace(lines[1])
	if keyFile != "" && commandLineContains(pid, keyFile) {
		logger.Warn("stopping engine left by a previous agent", "pid", pid, "replica", replica)
		stopPID(pid, 25*time.Second)
	}
	_ = os.Remove(pidFile)
}
