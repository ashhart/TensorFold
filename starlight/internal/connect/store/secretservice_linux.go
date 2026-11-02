//go:build linux

package store

import (
	"errors"
	"slices"
	"time"

	"github.com/godbus/dbus/v5"
)

const (
	ssName       = "org.freedesktop.secrets"
	ssPath       = dbus.ObjectPath("/org/freedesktop/secrets")
	ssService    = "org.freedesktop.Secret.Service"
	ssCollection = "org.freedesktop.Secret.Collection"
	ssItem       = "org.freedesktop.Secret.Item"
	ssPrompt     = "org.freedesktop.Secret.Prompt"
)

type secretValue struct {
	Session     dbus.ObjectPath
	Parameters  []byte
	Value       []byte
	ContentType string
}

// SecretService stores secrets in the freedesktop Secret Service, such as GNOME Keyring or KWallet.
type SecretService struct {
	conn *dbus.Conn
}

func openPlatform() (Store, error) {
	conn, err := dbus.SessionBus()
	if err != nil {
		return nil, err
	}
	var owned bool
	if err := conn.BusObject().Call("org.freedesktop.DBus.NameHasOwner", 0, ssName).Store(&owned); err != nil {
		return nil, err
	}
	if !owned {
		var names []string
		if err := conn.BusObject().Call("org.freedesktop.DBus.ListActivatableNames", 0).Store(&names); err != nil || !slices.Contains(names, ssName) {
			return nil, errors.New("no secret service on the session bus")
		}
	}
	return &SecretService{conn: conn}, nil
}

// Backend names the store for status output.
func (s *SecretService) Backend() string { return "Secret Service" }

func (s *SecretService) service() dbus.BusObject { return s.conn.Object(ssName, ssPath) }

func attributes(key string) map[string]string {
	return map[string]string{"application": "tensorfold-connect", "key": key}
}

func (s *SecretService) openSession() (dbus.ObjectPath, func(), error) {
	var out dbus.Variant
	var path dbus.ObjectPath
	if err := s.service().Call(ssService+".OpenSession", 0, "plain", dbus.MakeVariant("")).Store(&out, &path); err != nil {
		return "", nil, err
	}
	return path, func() { s.conn.Object(ssName, path).Call("org.freedesktop.Secret.Session.Close", 0) }, nil
}

func (s *SecretService) find(key string) (dbus.ObjectPath, error) {
	var unlocked, locked []dbus.ObjectPath
	if err := s.service().Call(ssService+".SearchItems", 0, attributes(key)).Store(&unlocked, &locked); err != nil {
		return "", err
	}
	if len(unlocked) > 0 {
		return unlocked[0], nil
	}
	if len(locked) > 0 {
		return locked[0], s.unlock(locked[:1])
	}
	return "", ErrNotFound
}

func (s *SecretService) unlock(objects []dbus.ObjectPath) error {
	var unlocked []dbus.ObjectPath
	var prompt dbus.ObjectPath
	if err := s.service().Call(ssService+".Unlock", 0, objects).Store(&unlocked, &prompt); err != nil {
		return err
	}
	return s.prompt(prompt)
}

// prompt runs a Secret Service prompt, such as a keyring unlock dialog, and waits for it to complete.
func (s *SecretService) prompt(p dbus.ObjectPath) error {
	if p == "" || p == "/" {
		return nil
	}
	match := []dbus.MatchOption{dbus.WithMatchObjectPath(p), dbus.WithMatchInterface(ssPrompt), dbus.WithMatchMember("Completed")}
	if err := s.conn.AddMatchSignal(match...); err != nil {
		return err
	}
	defer s.conn.RemoveMatchSignal(match...)
	signals := make(chan *dbus.Signal, 4)
	s.conn.Signal(signals)
	defer s.conn.RemoveSignal(signals)
	if err := s.conn.Object(ssName, p).Call(ssPrompt+".Prompt", 0, "").Err; err != nil {
		return err
	}
	timeout := time.NewTimer(2 * time.Minute)
	defer timeout.Stop()
	for {
		select {
		case sig := <-signals:
			if sig == nil || sig.Path != p || sig.Name != ssPrompt+".Completed" {
				continue
			}
			if len(sig.Body) > 0 {
				if dismissed, ok := sig.Body[0].(bool); ok && dismissed {
					return errors.New("the keyring prompt was dismissed")
				}
			}
			return nil
		case <-timeout.C:
			return errors.New("timed out waiting for the keyring prompt")
		}
	}
}

func (s *SecretService) collection() (dbus.ObjectPath, error) {
	var path dbus.ObjectPath
	if err := s.service().Call(ssService+".ReadAlias", 0, "default").Store(&path); err != nil {
		return "", err
	}
	if path == "/" {
		return "", errors.New("the secret service has no default collection")
	}
	if v, err := s.conn.Object(ssName, path).GetProperty(ssCollection + ".Locked"); err == nil {
		if locked, ok := v.Value().(bool); ok && locked {
			if err := s.unlock([]dbus.ObjectPath{path}); err != nil {
				return "", err
			}
		}
	}
	return path, nil
}

// Get returns the value stored at key.
func (s *SecretService) Get(key string) ([]byte, error) {
	item, err := s.find(key)
	if err != nil {
		return nil, err
	}
	session, closeSession, err := s.openSession()
	if err != nil {
		return nil, err
	}
	defer closeSession()
	var secret secretValue
	if err := s.conn.Object(ssName, item).Call(ssItem+".GetSecret", 0, session).Store(&secret); err != nil {
		return nil, err
	}
	return secret.Value, nil
}

// Set stores value at key, replacing any earlier value.
func (s *SecretService) Set(key string, value []byte) error {
	coll, err := s.collection()
	if err != nil {
		return err
	}
	session, closeSession, err := s.openSession()
	if err != nil {
		return err
	}
	defer closeSession()
	props := map[string]dbus.Variant{
		"org.freedesktop.Secret.Item.Label":      dbus.MakeVariant(Service + " (" + key + ")"),
		"org.freedesktop.Secret.Item.Attributes": dbus.MakeVariant(attributes(key)),
	}
	secret := secretValue{Session: session, Parameters: []byte{}, Value: value, ContentType: "application/octet-stream"}
	var item, prompt dbus.ObjectPath
	if err := s.conn.Object(ssName, coll).Call(ssCollection+".CreateItem", 0, props, secret, true).Store(&item, &prompt); err != nil {
		return err
	}
	return s.prompt(prompt)
}

// Delete removes key; deleting a missing key succeeds.
func (s *SecretService) Delete(key string) error {
	item, err := s.find(key)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	var prompt dbus.ObjectPath
	if err := s.conn.Object(ssName, item).Call(ssItem+".Delete", 0).Store(&prompt); err != nil {
		return err
	}
	return s.prompt(prompt)
}
