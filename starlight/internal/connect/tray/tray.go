//go:build !notray

package tray

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"fyne.io/systray"

	"github.com/ashhart/TensorFold/starlight/internal/connect/bridge"
	"github.com/ashhart/TensorFold/starlight/internal/connect/core"
	"github.com/ashhart/TensorFold/starlight/internal/connect/login"
	"github.com/ashhart/TensorFold/starlight/internal/connect/setup"
	"github.com/ashhart/TensorFold/starlight/internal/connect/starlightapi"
	"github.com/ashhart/TensorFold/starlight/internal/connect/tokens"
)

const maxModels = 24

type tray struct {
	core      *core.Core
	ctx       context.Context
	cancel    context.CancelFunc
	signingIn atomic.Bool
	mu        sync.Mutex
	bridge    *bridge.Server
	console   string

	identity, org, models, endpoint, openConsole, signIn, signOut, quit *systray.MenuItem
	copies                                                              map[string]*systray.MenuItem
	modelItems                                                          [maxModels]*systray.MenuItem
}

// Run shows the tray icon and blocks until the user quits or ctx ends.
func Run(parent context.Context, c *core.Core) error {
	ctx, cancel := context.WithCancel(parent)
	t := &tray{core: c, ctx: ctx, cancel: cancel, copies: map[string]*systray.MenuItem{}}
	systray.Run(t.ready, t.exit)
	return nil
}

func (t *tray) ready() {
	systray.SetIcon(Icon())
	systray.SetTooltip("TensorFold Connect")
	t.identity = systray.AddMenuItem("Not signed in", "")
	t.identity.Disable()
	t.org = systray.AddMenuItem("", "")
	t.org.Disable()
	t.org.Hide()
	t.models = systray.AddMenuItem("Models", "Models your organisation lets you use")
	for i := range t.modelItems {
		t.modelItems[i] = t.models.AddSubMenuItem("", "")
		t.modelItems[i].Disable()
		t.modelItems[i].Hide()
	}
	systray.AddSeparator()
	for _, c := range []struct{ tool, title string }{{"claude-code", "Copy setup for Claude Code"}, {"codex", "Copy setup for Codex"},
		{"openai", "Copy setup for OpenAI SDKs"}} {
		t.copies[c.tool] = systray.AddMenuItem(c.title, "Copies environment settings with a fresh app key")
	}
	t.endpoint = systray.AddMenuItemCheckbox("Local endpoint", t.core.BridgeURL(0), false)
	systray.AddSeparator()
	t.openConsole = systray.AddMenuItem("Open console", "")
	t.signIn = systray.AddMenuItem("Sign in…", "")
	t.signOut = systray.AddMenuItem("Sign out", "")
	systray.AddSeparator()
	t.quit = systray.AddMenuItem("Quit TensorFold Connect", "")
	if t.core.Settings.BridgeDisabled {
		t.endpoint.SetTitle("Local endpoint (turned off by your organisation)")
		t.endpoint.Disable()
	}
	for tool, item := range t.copies {
		go t.watchCopy(tool, item)
	}
	go t.loop()
}

func (t *tray) loop() {
	t.refresh()
	if !t.core.Settings.BridgeDisabled {
		t.toggleBridge()
	}
	tick := time.NewTicker(60 * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-t.ctx.Done():
			systray.Quit()
			return
		case <-tick.C:
			t.refresh()
		case <-t.endpoint.ClickedCh:
			t.toggleBridge()
		case <-t.openConsole.ClickedCh:
			if t.console != "" {
				_ = login.OpenBrowser(t.console)
			}
		case <-t.signIn.ClickedCh:
			go t.runSignIn()
		case <-t.signOut.ClickedCh:
			ctx, cancel := context.WithTimeout(t.ctx, 15*time.Second)
			_ = t.core.Logout(ctx)
			cancel()
			t.refresh()
		case <-t.quit.ClickedCh:
			systray.Quit()
			return
		}
	}
}

func (t *tray) exit() {
	t.mu.Lock()
	b := t.bridge
	t.bridge = nil
	t.mu.Unlock()
	if b != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_ = b.Shutdown(ctx)
		cancel()
	}
	t.cancel()
}

func (t *tray) refresh() {
	ctx, cancel := context.WithTimeout(t.ctx, 20*time.Second)
	defer cancel()
	server, err := t.core.Server()
	if err != nil {
		t.showSignedOut("No Starlight server configured")
		return
	}
	systray.SetTooltip("TensorFold Connect · " + server)
	if doc, err := t.core.Discovery(ctx); err == nil {
		t.console = doc.ConsoleURL
	}
	api, err := t.core.API(ctx)
	if err != nil {
		t.showSignedOut("Starlight unreachable")
		return
	}
	me, err := api.Me(ctx)
	if errors.Is(err, tokens.ErrSignedOut) {
		t.showSignedOut("Not signed in")
		return
	}
	if err != nil {
		t.identity.SetTitle("Starlight unreachable; retrying")
		return
	}
	t.identity.SetTitle("Signed in as " + me.Email)
	org := me.OrgName
	if p := t.core.Settings.Project; p != "" {
		org += " · " + p
	}
	t.org.SetTitle(org)
	t.org.Show()
	t.signIn.Hide()
	t.signOut.Show()
	for _, item := range t.copies {
		item.Enable()
	}
	models, err := api.Models(ctx, t.core.Settings.Project)
	if err != nil {
		return
	}
	t.showModels(models)
}

func (t *tray) showModels(models []starlightapi.Model) {
	t.models.SetTitle(fmt.Sprintf("Models (%d)", len(models)))
	for i, item := range t.modelItems {
		switch {
		case i < len(models) && (i < maxModels-1 || len(models) == maxModels):
			item.SetTitle(models[i].ID)
			item.Show()
		case i == maxModels-1 && len(models) > maxModels:
			item.SetTitle(fmt.Sprintf("and %d more", len(models)-(maxModels-1)))
			item.Show()
		default:
			item.Hide()
		}
	}
	if len(models) == 0 {
		t.modelItems[0].SetTitle("No models available to you")
		t.modelItems[0].Show()
	}
}

func (t *tray) showSignedOut(reason string) {
	t.identity.SetTitle(reason)
	t.org.Hide()
	t.signIn.Show()
	t.signOut.Hide()
	for _, item := range t.modelItems {
		item.Hide()
	}
	t.models.SetTitle("Models")
}

func (t *tray) runSignIn() {
	if !t.signingIn.CompareAndSwap(false, true) {
		return
	}
	defer t.signingIn.Store(false)
	t.identity.SetTitle("Finish signing in in your browser…")
	ctx, cancel := context.WithTimeout(t.ctx, 6*time.Minute)
	defer cancel()
	if _, err := t.core.Login(ctx, core.LoginOptions{Open: login.OpenBrowser, Out: io.Discard}); err != nil {
		t.identity.SetTitle("Sign-in failed: " + short(err))
		return
	}
	t.refresh()
}

func (t *tray) watchCopy(tool string, item *systray.MenuItem) {
	title := map[string]string{"claude-code": "Copy setup for Claude Code", "codex": "Copy setup for Codex", "openai": "Copy setup for OpenAI SDKs"}[tool]
	for {
		select {
		case <-t.ctx.Done():
			return
		case <-item.ClickedCh:
		}
		text, replaced, err := t.core.Setup(tool, "", "", setup.DefaultShell(), 0)
		if err == nil {
			err = CopyText(text)
		}
		switch {
		case err != nil:
			item.SetTitle("Could not copy: " + short(err))
		case replaced:
			item.SetTitle("Copied; the previous " + tool + " key no longer works")
		default:
			item.SetTitle("Copied to the clipboard")
		}
		time.AfterFunc(4*time.Second, func() { item.SetTitle(title) })
	}
}

func (t *tray) toggleBridge() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.bridge != nil {
		ctx, cancel := context.WithTimeout(t.ctx, 5*time.Second)
		_ = t.bridge.Shutdown(ctx)
		cancel()
		t.bridge = nil
		t.endpoint.Uncheck()
		t.endpoint.SetTitle("Local endpoint")
		return
	}
	ctx, cancel := context.WithTimeout(t.ctx, 20*time.Second)
	defer cancel()
	b, err := t.core.Bridge(ctx, 0)
	if err == nil {
		err = b.Start()
	}
	if err != nil {
		t.endpoint.SetTitle("Local endpoint: " + short(err))
		return
	}
	t.bridge = b
	t.endpoint.Check()
	t.endpoint.SetTitle("Local endpoint on " + b.URL())
}

func short(err error) string {
	s := err.Error()
	if len(s) > 60 {
		s = s[:57] + "..."
	}
	return s
}
