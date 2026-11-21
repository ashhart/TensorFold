# ComputerUseChecks

Run the shared parser, key-table, phone-input validation and JSON-line checks with Swift 6.2 or later on macOS or Linux:

```sh
swift test --package-path app/native/computer-use/ComputerUseChecks
```

The Mac helper compiles these same `Sources/ComputerUseChecks` files directly. Package tests, the manifest and `.build` artifacts are excluded from that binary. Linux uses Foundation geometry and small key-code/event-flag stand-ins; macOS uses CoreGraphics.

Click execution, screen inspection, native phone-window checks and OCR stay in the Mac helper. From `app/`, build it and run its remaining self-checks with `npm run build:computer-helper`.
