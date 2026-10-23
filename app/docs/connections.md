# Google and messaging connections

Open Settings, Assistant, Connectors and choose the service to configure.
The Google connector ships in the beta.9 installer and later; it is still
pending qualification against a real account.

## Gmail and Google Calendar

Each installation uses its own Google Cloud project and Desktop OAuth client.
TensorFold does not require a shared publisher OAuth project.

1. Create your project and configure the OAuth consent audience in
   [Google Cloud](https://console.cloud.google.com/auth/overview).
2. Enable Gmail API and Google Calendar API for the services you want.
3. Create a **Desktop app** OAuth client and download its JSON file.
4. Import that JSON through the native file chooser in TensorFold.
5. Choose access, open Google's browser sign-in and select the connected account.

For a project in Testing, add the intended account as a test user; Google may
require reconnection under its testing policies.
BYO credentials do not remove Google's consent, organization or verification
requirements, which depend on the project's audience and requested scopes.
See Google's [native app OAuth documentation](https://developers.google.com/identity/protocols/oauth2/native-app)
and [Gmail scope classifications](https://developers.google.com/workspace/gmail/api/auth/scopes).
Do not share client JSON, app passwords or account tokens in a chat or repository.

### Troubleshooting Google sign-in

- **The client must be a Desktop app client.** A web client or a service-account
  key cannot sign in; Google rejects it and the import step says so.
- **A project in Testing needs test users.** Add the Google account you are
  signing in with to the project's test users, or Google refuses the sign-in
  with "access denied" even though you approved the permissions.
- **The Gmail and Calendar APIs must be enabled.** This is not checked at
  sign-in: the sign-in succeeds and the failure appears later, when the app
  exchanges the authorization code for a token or calls the API.
- **What the browser says.** "Google sign-in received. Return to TensorFold to
  check the connection; you can close this tab." means the sign-in reached the
  app. "Sign-in was not completed. You can close this tab and return to
  TensorFold." means Google refused it, and the app shows what to check.
  "Sign-in was not completed. Return to TensorFold for details." means the
  callback reached the app but could not be used, and the app names the
  problem. "Invalid sign-in callback. Return to TensorFold and try again." is
  not your callback at all — a favicon, another address or a request without
  our sign-in token — and nothing was signed in.
- **Retrying.** A failed or declined sign-in clears the waiting state at once.
  The sign-in button is available again without restarting the app, and each
  attempt is a new sign-in with a new one-time token.

Read Gmail and Read Calendar are selected initially; sends and Calendar writes
need separate permission choices.
Adding access opens a new consent flow for the desired permission set, and
TensorFold checks the access Google actually granted.
Credentials stay in the host's encrypted protected store and never enter the
renderer or model context; locked or invalid stores fail closed.

Search/read does not change Gmail read status, load tracking images or download
attachments automatically.
Email drafts are local TensorFold drafts bound to one account, not Gmail drafts.
Every email send and Calendar mutation needs a complete saved review followed by
native owner confirmation on the Mac.
Updates and cancellations check the current event version before execution.
All-day ends are exclusive, and ambiguous daylight-saving times need an explicit
offset choice.
Guest notification choices are visible, but Google may still send some notices
when no updates are requested.

Action outcomes survive restart; uncertain outcomes are never automatically retried.
Check Google before preparing a replacement for an uncertain operation.
Disconnect removes local credentials/drafts/reviews and attempts grant revocation;
Google revocation affects all services and clients sharing that project/account grant.

Google tools require an explicitly bound API model running on this Mac.
Conversations retaining Google-derived content remain local-only across restarts,
edits and supported branches, and are excluded from other-chat recall, automatic
memory harvesting, remote/watch projections and automatic speech.
They cannot move to cloud/LAN/CLI/remote providers or launch linked child agents.
Google content is not exposed through Telegram Chief of Staff in this version.

## Slack and Discord

Use bot tokens from the services' official developer consoles, not personal user
tokens; save and verify each connection before choosing its channels.
Empty channel lists grant no access.
Use `*` alone only to explicitly grant every channel the bot can access, including
future channels, then connect and request a sync.
New connections start disabled, and source-specific verification/sync errors stay
visible beside that connection.
Slack export folders remain supported separately.

Slack bots have restrictions on public/private thread history; ordinary channel
history requires the appropriate scopes and channel membership.
Discord channel permissions, Message Content Intent and thread permissions still
apply; Administrator permission is unnecessary.
API rate limits and workspace/server policies may restrict reads.

## Telegram remote assistant

Create a bot with Telegram's verified BotFather, save/verify its token, then enter
explicit numeric chat and sender allowlists; both must match before commands run.
Use your own bot's updates to obtain IDs, not third-party ID bots.
Background polling is optional; no public webhook is required.

Enable remote access separately and select the chats/rooms it may access.
The default grants no conversations; all eligible current/future chats is an
explicit broader choice.
Remote mode accepts only an allowlisted owner's private chat with matching sender
and chat IDs; groups cannot use it.
Incognito, private, phone-confined, Google-history, archived, remote and unsupported
bot conversations are excluded by the host.

`/chief` opens the local read-only Chief of Staff overview; `/chats` and `/rooms`
list allowed targets, while `/chat id`, `/room id` and `/back` change the selection.
In Chief mode, `dispatch chat id your request` explicitly sends a task to an allowed
chat; model output cannot choose execution targets.
Chats reuse their existing model, workspace and approval rules, busy chats refuse
new work, and room guidance only goes to active allowed rooms.
Telegram cannot approve actions that require the Mac; queued/incomplete replies
tell you to inspect the retained run there.
Revoking access or rotating a token invalidates in-flight authorization.

Keep the Mac awake and TensorFold running for polling and local inference.
Telegram bot conversations are not end-to-end encrypted.
Real Google OAuth, Gmail/Calendar operations and live Slack/Discord/Telegram
accounts still require qualification with the intended client's configuration
before a release can claim those live flows are verified.
