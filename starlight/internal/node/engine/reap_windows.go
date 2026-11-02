package engine

import "time"

// commandLineContains is never needed on Windows: the kill-on-close job object ends orphans with the agent.
func commandLineContains(int, string) bool { return false }

func stopPID(int, time.Duration) {}
