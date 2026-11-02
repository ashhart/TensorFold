package login

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/oauth"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

const deviceGrant = "urn:ietf:params:oauth:grant-type:device_code"

// Device runs the RFC 8628 device grant: the user approves the shown code on any browser.
func Device(ctx context.Context, doc discovery.Document, o Options) (discovery.TokenResponse, error) {
	o.defaults()
	form := url.Values{"client_id": {discovery.ClientConnect}, "scope": {oauth.Scope}}
	if o.DeviceName != "" {
		form.Set("device_name", o.DeviceName)
	}
	var dr discovery.DeviceResponse
	if err := oauth.Post(ctx, o.Client, doc.DeviceAuthorizationEndpoint, form, &dr); err != nil {
		return discovery.TokenResponse{}, err
	}
	if dr.DeviceCode == "" || dr.UserCode == "" || dr.VerificationURI == "" {
		return discovery.TokenResponse{}, errors.New("device authorization reply is incomplete")
	}
	fmt.Fprintf(o.Out, "To sign in, open %s on any device and enter the code:\n\n  %s\n\n", dr.VerificationURI, dr.UserCode)
	if dr.VerificationURIComplete != "" {
		fmt.Fprintf(o.Out, "Or open %s\n\n", dr.VerificationURIComplete)
	}
	interval := dr.Interval
	if interval <= 0 {
		interval = 5
	}
	expires := dr.ExpiresIn
	if expires <= 0 {
		expires = 600
	}
	deadline := time.Now().Add(time.Duration(expires) * o.unit)
	poll := url.Values{"grant_type": {deviceGrant}, "device_code": {dr.DeviceCode}, "client_id": {discovery.ClientConnect}}
	for {
		wait := time.NewTimer(time.Duration(interval) * o.unit)
		select {
		case <-ctx.Done():
			wait.Stop()
			return discovery.TokenResponse{}, ctx.Err()
		case <-wait.C:
		}
		if time.Now().After(deadline) {
			return discovery.TokenResponse{}, errors.New("the device code expired before it was approved; run login again")
		}
		tr, err := oauth.Token(ctx, o.Client, doc.TokenEndpoint, poll)
		switch {
		case err == nil:
			return tr, nil
		case oauth.Is(err, "authorization_pending"):
		case oauth.Is(err, "slow_down"):
			interval += 5
		case oauth.Is(err, "access_denied"):
			return discovery.TokenResponse{}, errors.New("sign-in was denied")
		case oauth.Is(err, "expired_token"):
			return discovery.TokenResponse{}, errors.New("the device code expired before it was approved; run login again")
		default:
			return discovery.TokenResponse{}, err
		}
	}
}
