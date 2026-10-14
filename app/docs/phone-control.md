# Phone control

Forks retain a phone conversation's local-model restriction without inheriting its device grant.
Handing off copied phone context is limited to API models on this Mac; remote, external and CLI targets are refused before the transcript is prepared.
Phone-local conversations cannot launch child agents or linked missions, whose independent runtimes do not yet enforce the conversation's local-processing boundary.
Phone-local runs also use an empty plugin runtime and suppress automatic plugin, observer and compaction-harvest callbacks, including when a run becomes phone-local during observation.
The Phone panel and encrypted conversation retain the native workflow; automatic shared tool-output spills are disabled for phone-local output.

TensorFold's original phone integration connects one device to a local chat through its Phone
panel. It uses platform APIs and protocols, with no phone-harness code, dependency or installer.

## Android

Install [Google Android Platform Tools](https://developer.android.com/tools/releases/platform-tools)
and select its adb executable in Settings → Phone control if it is not detected in a standard SDK
location. Use a USB data cable, enable Developer options and USB debugging, and approve the
computer on the phone. Unauthorized or offline devices stay visible with setup guidance.

For Android 11 or later Wireless debugging, pair with the numeric private address, pairing port
and six-digit code shown on the phone, then connect with its separate connection port. Pairing
codes go through stdin, are cleared from the form after each attempt and are never saved in
settings or logs. TensorFold does not enable legacy unauthenticated tcpip mode, scan the LAN or
stop the shared ADB server. Closing TensorFold does not revoke a pairing another application uses.

The adapter checks lock state before reading UI, refuses unknown lock states and recognized
password screens, and uses a private temporary UI dump with targeted cleanup. Screenshots are
bounded and validated; commands pin the selected serial and use fixed operations and arguments.
Plain ASCII typing is supported, with literal %s rejected because Android input treats it as
space; use the phone keyboard for other scripts or this literal sequence.

## iPhone

[Apple iPhone Mirroring](https://support.apple.com/en-gb/120421) requires a compatible Mac and
iPhone, the same Apple Account with two-factor authentication, Wi-Fi and Bluetooth, and a nearby
locked phone. Pair in Apple's app first, then explicitly check Accessibility and Screen Recording
permissions in TensorFold Settings. TensorFold pins Apple's process and exact phone window,
captures only its content and raises that pinned window for requested input; there is no desktop
capture or arbitrary-app fallback. OCR provides text, so unknown icons or controls need manual use.
Availability depends on Apple's regional and device requirements.

USB-C alone does not expose an iPhone's general UI automation API. The developer bridge option
requires your own signed and running [WebDriverAgent](https://appium.github.io/appium-xcuitest-driver/latest/preparation/real-device-config/), compatible Xcode device support, Developer Mode and
phone trust. TensorFold neither installs this runner nor changes phone security. For USB, use an
existing loopback-only port forward to its port, for example with an installed libusbmuxd iproxy:

```sh
iproxy -l -s 127.0.0.1 -u PAIRED_DEVICE_ID 8100:8100
```

Enter http://127.0.0.1:8100 in TensorFold's USB bridge setup. For the developer Wi-Fi route, enter
the signed runner's numeric private IP and port on a trusted private network. Public/DNS targets,
credentials in URLs, redirects and arbitrary URL paths are refused. WebDriverAgent is a developer
endpoint, so protect the phone's network and stop the runner when you finish. TensorFold creates
and deletes only its own session and refuses a runner already controlled by another client.

## Chat permissions and privacy

Connection is separate from allowing a chat. Allow this chat requires a model running on this
Mac, a local chat rather than SSH/orchestrator work, and Cortheon disabled. A durable local-only
marker protects that conversation's derived phone context across disconnects and app restarts;
start a separate chat for cloud models or web research. Other chats cannot recall its messages,
summaries or linked artifacts through session search, session reads or workspace retrieval.
The original phone chat can still use its own encrypted history. Each agent input needs review against
its exact session and observation IDs. Screens are re-read and compared immediately before input,
including after a delayed approval, and read again afterward. Only explicit Home/Back navigation
can tolerate changed pixels when the known foreground app, dimensions and normalized accessible
controls still match. Taps, swipes, typing, launching apps and manual Enter keep complete screenshot
binding; an unknown app gets no pixel-drift exception. Navigation without a semantic change still
counts toward the three-action pause even if status pixels animate. Manual preview inputs expire after
30 seconds. A session is limited to 40 dispatched agent actions and 15 minutes per grant, with
three unchanged results pausing control. Stop aborts input, cancels the owning chat run and clears
the preview; normal Quit waits for phone cleanup as well as any learning checkpoint save.

Screenshots are not written to disk or included in model messages by this feature. Accessible
labels used by an allowed model become part of that local chat and use its existing encrypted
persistence. Automatic extension hooks, auxiliary observers and automatic tool-output spill files
do not receive newly captured phone context, including when a run becomes phone-local after it starts.
Recognized secure fields and locks are refused before capture; model taps require
a labelled interactive control or conservative OCR navigation label, and typing requires a
verified focused nonsecure text field. Sending, payments, authentication, deletion and access
confirmation stay with the owner. Arbitrary custom secure controls, every language and every
vendor UI cannot be identified perfectly; the integration does not claim universal PII detection.

## Qualification

Automated providers use synthetic process/server/helper fixtures, and the chat integration is
qualified separately against an account-free Android emulator and a local model. This does not
qualify physical USB/Wi-Fi hardware, Apple Mirroring geometry or signed iPhone runner setup on
every phone/OS version. Those remain explicit physical-device release checks.
