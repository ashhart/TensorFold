package service

import (
	"context"
	"time"

	"golang.org/x/sys/windows/svc"
)

// IsService reports whether the service control manager started this process.
func IsService() bool {
	ok, err := svc.IsWindowsService()
	return err == nil && ok
}

// RunService hands the process to the service control manager and runs fn until Stop or Shutdown.
func RunService(fn func(context.Context) error) error {
	return svc.Run(WindowsName, &handler{fn: fn})
}

type handler struct {
	fn func(context.Context) error
}

func (h *handler) Execute(_ []string, req <-chan svc.ChangeRequest, st chan<- svc.Status) (bool, uint32) {
	st <- svc.Status{State: svc.StartPending}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- h.fn(ctx) }()
	accepts := svc.AcceptStop | svc.AcceptShutdown
	st <- svc.Status{State: svc.Running, Accepts: accepts}
	for {
		select {
		case err := <-done:
			if err != nil {
				return true, 1
			}
			return false, 0
		case c := <-req:
			switch c.Cmd {
			case svc.Interrogate:
				st <- c.CurrentStatus
			case svc.Stop, svc.Shutdown:
				cancel()
				tick := time.NewTicker(10 * time.Second)
				for checkpoint := uint32(1); ; checkpoint++ {
					st <- svc.Status{State: svc.StopPending, CheckPoint: checkpoint, WaitHint: 30000}
					select {
					case <-done:
						tick.Stop()
						return false, 0
					case <-tick.C:
					}
				}
			}
		}
	}
}
