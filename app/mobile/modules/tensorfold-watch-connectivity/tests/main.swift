import Foundation
import Security

var assertions = 0
func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
  assertions += 1
  guard condition() else { fatalError(message) }
}
func rejects(_ message: String, _ action: () throws -> Void) {
  do { try action(); fatalError("Expected rejection: " + message) } catch { assertions += 1 }
}
let fixture = try WatchBackgroundConfiguration(baseUrl: "https://mac.example:9999/private/", token: "fixture-phone-only-token", sessionId: "session-test-1", certificatePin: String(repeating: "A", count: 64))
expect(fixture.endpoint.absoluteString == "https://mac.example:9999/private/v1/watch/command", "The fixed endpoint preserves the paired base path")
expect(fixture.certificatePin == String(repeating: "a", count: 64), "Pin normalization")
for address in ["http://localhost:9999", "https://user:pass@host", "https://host?token=x", "https://host#token", "https://host/../escape", "https://host/%2e%2e/escape"] {
  rejects("unsafe base URL") { _ = try WatchBackgroundConfiguration(baseUrl: address, token: fixture.token, sessionId: fixture.sessionId, certificatePin: fixture.certificatePin) }
}
rejects("header injection") { _ = try WatchBackgroundConfiguration(baseUrl: fixture.baseUrl, token: "fixture-token\r\nInjected: yep", sessionId: fixture.sessionId, certificatePin: fixture.certificatePin) }
rejects("missing pin") { _ = try WatchBackgroundConfiguration(baseUrl: fixture.baseUrl, token: fixture.token, sessionId: fixture.sessionId, certificatePin: "") }
func payload(_ action: String = "refresh", id: String = UUID().uuidString, extras: [String: Any] = [:]) -> String {
  var value: [String: Any] = ["id": id, "version": 1, "session": fixture.sessionId, "action": action, "generatedAt": 1_000_000]
  extras.forEach { value[$0.key] = $0.value }
  return String(data: try! JSONSerialization.data(withJSONObject: value), encoding: .utf8)!
}
for extras: [String: Any] in [["session": "different-session"], ["version": true], ["generatedAt": true], ["generatedAt": 900_000], ["token": fixture.token], ["action": "arbitrary-api-call"]] {
  rejects("invalid bounded command") { _ = try WatchBackgroundCommand(payload: payload(extras: extras), session: fixture.sessionId, now: 1_000) }
}
rejects("oversized UTF16 reply") { _ = try WatchBackgroundCommand(payload: payload("reply", extras: ["threadId": "thread-1", "text": String(repeating: "😀", count: 1_001)]), session: fixture.sessionId, now: 1_000) }
rejects("noninteger audio sequence") { _ = try WatchBackgroundCommand(payload: payload("audio-next", extras: ["speechId": "speech-id-1", "sequence": 1.5]), session: fixture.sessionId, now: 1_000) }
for action in ["refresh", "reply", "audio-start", "audio-next", "audio-stop"] {
  let command = try WatchBackgroundCommand(payload: payload(action, extras: ["threadId": "thread-1", "text": "Hello", "itemId": "assistant-1", "speechId": "speech-id-1", "sequence": 0]), session: fixture.sessionId, now: 1_000)
  expect(command.action == action, "Every supported operation reaches the fixed host endpoint")
  expect(!String(data: command.payload, encoding: .utf8)!.contains(fixture.token), "Phone credential never enters watch command")
}
final class Task: WatchBackgroundCancellable { var cancelled = false; func cancel() { cancelled = true } }
final class Budget: WatchBackgroundBudget { var ended = 0; let expire: () -> Void; init(_ expire: @escaping () -> Void) { self.expire = expire }; func end() { ended += 1 } }
var tasks: [Task] = [], budgets: [Budget] = [], callbacks: [(Result<Data, Error>) -> Void] = [], commands: [WatchBackgroundCommand] = []
let relay = WatchBackgroundRelay(transport: { configuration, command, callback in
  expect(configuration.token == fixture.token, "Only native transport gets credential")
  let task = Task(); tasks.append(task); callbacks.append(callback); commands.append(command); return task
}, budget: { callback in let budget = Budget(callback); budgets.append(budget); return budget }, now: { 1_000 })
var results: [[String: Any]] = []
relay.handle(payload()) { results.append($0) }
expect(tasks.isEmpty && results.last?["error"] != nil, "No consent means no network")
relay.configure(fixture)
for _ in 0..<5 { relay.handle(payload("reply", extras: ["threadId": "thread-1", "text": "Hello"])) { results.append($0) } }
expect(tasks.count == 4 && budgets.count == 4, "Four request concurrency cap")
expect(results.last?["ok"] as? Bool == false && results.last?["uncertain"] == nil, "Capacity refusal is certain before dispatch")
budgets[0].expire()
expect(tasks[0].cancelled && budgets[0].ended == 1 && results.last?["uncertain"] as? Bool == true, "Expiry cancels delivery and exposes uncertainty")
let delivered = try JSONSerialization.data(withJSONObject: ["id": commands[0].id, "ok": true, "message": "accepted"])
let count = results.count; callbacks[0](.success(delivered))
expect(results.count == count, "Late completion after expiration is suppressed")
relay.configure(nil)
expect(tasks.dropFirst().allSatisfy { $0.cancelled } && budgets.allSatisfy { $0.ended == 1 }, "Opt out cancels pending requests and ends every budget")
let revokedCount = results.count
for index in 1..<callbacks.count { callbacks[index](.success(try JSONSerialization.data(withJSONObject: ["id": commands[index].id, "ok": true, "message": "stale"]))) }
expect(results.count == revokedCount, "Revoked-session results never reach watch")
relay.configure(fixture)
relay.handle(payload()) { results.append($0) }
callbacks.last?(.success(try JSONSerialization.data(withJSONObject: ["id": commands.last!.id, "ok": true, "message": "Updated", "snapshot": ["session": fixture.sessionId]])))
expect(results.last?["ok"] as? Bool == true && budgets.last?.ended == 1, "Successful result ends budget once")
relay.handle(payload("reply", extras: ["threadId": "thread-1", "text": "Hello"])) { results.append($0) }
callbacks.last?(.success(try JSONSerialization.data(withJSONObject: ["id": "wrong-response-id", "ok": true, "message": "invalid"])))
expect(results.last?["uncertain"] as? Bool == true && results.last?["ok"] as? Bool == false, "Mismatched response cannot confirm mutation")
let noBudget = WatchBackgroundRelay(transport: { _, _, _ in fatalError("No network without execution budget") }, budget: { _ in nil }, now: { 1_000 })
noBudget.configure(fixture); noBudget.handle(payload()) { expect($0["ok"] as? Bool == false, "Denied budget fails explicitly") }
let synchronousBudget = Budget({})
let expiredBeforeTransport = WatchBackgroundRelay(transport: { _, _, _ in fatalError("No network after immediate expiration") }, budget: { expire in expire(); return synchronousBudget }, now: { 1_000 })
expiredBeforeTransport.configure(fixture); expiredBeforeTransport.handle(payload()) { _ in }
expect(synchronousBudget.ended == 1, "Immediate budget expiration cannot leak its allocation")

final class FakeKeychain: WatchBackgroundKeychain {
  var data: Data?; var locked = false; var deleted = 0; var accessibility: String?; var failWrite = false
  func delete(_ query: [String: Any]) -> OSStatus { if locked { return errSecInteractionNotAllowed }; data = nil; deleted += 1; return errSecSuccess }
  func read(_ query: [String: Any]) -> (OSStatus, Data?) { locked ? (errSecInteractionNotAllowed, nil) : (data == nil ? errSecItemNotFound : errSecSuccess, data) }
  func update(_ query: [String: Any], _ attributes: [String: Any]) -> OSStatus {
    if locked || failWrite { return errSecInteractionNotAllowed }
    guard data != nil else { return errSecItemNotFound }
    data = attributes[kSecValueData as String] as? Data; accessibility = attributes[kSecAttrAccessible as String] as? String; return errSecSuccess
  }
  func add(_ attributes: [String: Any]) -> OSStatus {
    if locked || failWrite { return errSecInteractionNotAllowed }
    data = attributes[kSecValueData as String] as? Data; accessibility = attributes[kSecAttrAccessible as String] as? String; return errSecSuccess
  }
}
let suite = "dev.tensorfold.tests.watch." + UUID().uuidString, defaults = UserDefaults(suiteName: suite)!
defer { defaults.removePersistentDomain(forName: suite) }
let fake = FakeKeychain(), store = WatchBackgroundConfigurationStore(defaults: defaults, service: "fixture-private", keychain: fake)
fake.data = try JSONEncoder().encode(fixture)
expect(try! store.load() == nil, "Fresh install never resurrects Keychain consent")
expect(fake.deleted == 1, "Fresh install removes prior installation secret")
try store.save(fixture)
expect(fake.accessibility == kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly as String, "Watch credential is device-only, available only after first unlock")
expect(try! store.load() == fixture, "Explicit consent survives native cold start")
expect(!defaults.dictionaryRepresentation().values.contains { String(describing: $0).contains(fixture.token) }, "No secret in UserDefaults")
fake.locked = true
rejects("before first unlock") { _ = try store.load() }
expect(defaults.bool(forKey: "tensorfold.watch.background.consent.v1"), "Temporary first-unlock failure does not revoke or delete saved consent")
rejects("locked deletion") { try store.clear() }
expect(!defaults.bool(forKey: "tensorfold.watch.background.consent.v1"), "Failed Keychain deletion still revokes durable local gate")
fake.locked = false
expect(try! store.load() == nil, "Cold start cannot reactivate revoked consent")
try store.save(fixture); fake.failWrite = true
rejects("secure-write failure") { try store.save(fixture) }
expect(try! store.load() == nil, "Failed configuration remains disabled despite old secret")
print("PASS \(assertions) native background relay and consent assertions; fake network/Keychain, no device or account access")
