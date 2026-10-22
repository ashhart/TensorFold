# Google account recovery fixture

This page mounts the production Google Workspace connector, hooks, controls and
styles with a synthetic, in-memory preload. It never signs in to Google, reads
real accounts or uses credentials; external browser fetches/navigation are
blocked. The controls at the top belong to the fixture, not the product.

With the repository's pinned Node/npm and installed dependencies:

```sh
cd app
VONTRA_NO_ELECTRON=1 PORT=5188 npm run dev:web
```

Open `http://127.0.0.1:5188/experiments/google-account-recovery/`.
Scenarios cover no saved accounts, two accounts, a disconnected account, expiry
on the next Gmail read, a failed status read that recovers on Retry, and initial
Desktop OAuth client setup. Scenario changes deliberately reset the fixture.

To check expiry and draft preservation:

1. Choose **Expired on next Gmail read** and enter a synthetic recipient,
   subject and message in **Write an email**.
2. Press **Search** in Gmail. The account changes to **Reconnect needed**;
   its draft fields remain, saving is disabled, and account switching stays
   locked while there are unsaved edits.
3. Press **Reconnect with selected access**, then the fixture's **Finish fake
   sign-in**. The real hook polls the synthetic status to completion.
4. The same draft can now be saved locally; no email is sent.

The actual service invalid-grant test is separate:

```sh
node node_modules/jiti/lib/jiti-cli.mjs src/components/settings/connectors/google/GoogleAccountRecovery.test.tsx
```

It injects both transport and DNS lookup and checks that the synthetic token
exchange was used. Real Google consent, account policy and OS credential-store
qualification remain separate acceptance checks.
