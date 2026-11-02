package service

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

// Install registers an automatic-start service with restart-on-failure recovery and starts it.
func Install(o Options) (string, error) {
	if err := o.normalize(); err != nil {
		return "", err
	}
	if o.User != "" {
		return "", errors.New("the Windows service runs as LocalSystem; --user is not supported")
	}
	if o.DryRun {
		return fmt.Sprintf("service %s: %s %s", WindowsName, o.Binary, strings.Join(o.Args, " ")), nil
	}
	m, err := mgr.Connect()
	if err != nil {
		return "", fmt.Errorf("service manager (run as Administrator): %w", err)
	}
	defer m.Disconnect()
	if s, err := m.OpenService(WindowsName); err == nil {
		s.Close()
		return "", fmt.Errorf("service %s already exists; uninstall it first", WindowsName)
	}
	s, err := m.CreateService(WindowsName, o.Binary, mgr.Config{DisplayName: "TensorFold Starlight Node",
		Description: "Runs TensorFold engines assigned by TensorFold Starlight.", StartType: mgr.StartAutomatic}, o.Args...)
	if err != nil {
		return "", err
	}
	defer s.Close()
	actions := []mgr.RecoveryAction{{Type: mgr.ServiceRestart, Delay: 5 * time.Second}, {Type: mgr.ServiceRestart, Delay: 15 * time.Second},
		{Type: mgr.ServiceRestart, Delay: time.Minute}}
	if err := s.SetRecoveryActions(actions, 86400); err != nil {
		return "", err
	}
	if err := s.Start(); err != nil {
		return "", err
	}
	return "installed service " + WindowsName, nil
}

// Uninstall stops and deletes the service; state and models stay.
func Uninstall(string) error {
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()
	s, err := m.OpenService(WindowsName)
	if err != nil {
		return nil
	}
	defer s.Close()
	_, _ = s.Control(svc.Stop)
	return s.Delete()
}

// Status reports the service control manager's state.
func Status(string) (string, error) {
	m, err := mgr.Connect()
	if err != nil {
		return "", err
	}
	defer m.Disconnect()
	s, err := m.OpenService(WindowsName)
	if err != nil {
		return "not installed", nil
	}
	defer s.Close()
	st, err := s.Query()
	if err != nil {
		return "", err
	}
	names := map[svc.State]string{svc.Stopped: "stopped", svc.StartPending: "starting", svc.StopPending: "stopping", svc.Running: "running",
		svc.ContinuePending: "continuing", svc.PausePending: "pausing", svc.Paused: "paused"}
	return names[st.State], nil
}
