import Foundation
import WatchConnectivity

/// The singleton survives React bridge suspension/teardown. It starts from the
/// app delegate so a watch message can wake iOS without a running JS runtime.
final class WatchPhoneSession: NSObject, WCSessionDelegate {
  static let shared = WatchPhoneSession()
  private let store = WatchBackgroundConfigurationStore()
  private var started = false
  private var latest = ""
  private var restoreError: String?
  private lazy var relay: WatchBackgroundRelay = {
    let value = WatchBackgroundRelay(transport: { configuration, command, completion in
      let request = WatchBackgroundRequest(configuration: configuration, command: command, completion: completion)
      request.start(); return request
    }, budget: { WatchBackgroundExecutionBudget(expiration: $0) })
    value.onResult = { [weak self] result in self?.publishResultSnapshot(result) }
    return value
  }()

  func start() {
    guard !started else { return }; started = true
    restoreConfiguration()
    guard WCSession.isSupported() else { return }
    WCSession.default.delegate = self; WCSession.default.activate()
  }
  private func restoreConfiguration() {
    do {
      relay.configure(try store.load()); restoreError = nil
      if relay.configuration != nil, latest.isEmpty { publishBootstrap() }
    }
    catch let error as WatchBackgroundFailure { restoreError = error.message }
    catch { restoreError = "Background Apple Watch access could not be restored; enable it again in TensorFold on your iPhone." }
  }
  func backgroundStatus() -> [String: Any] {
    start()
    if relay.configuration == nil { restoreConfiguration() }
    var status: [String: Any] = ["configured": relay.configuration != nil]
    if let configuration = relay.configuration { status["sessionId"] = configuration.sessionId }
    if let restoreError { status["reason"] = restoreError }
    return status
  }
  func configureBackground(baseUrl: String, token: String, sessionId: String, certificatePin: String) throws {
    start()
    let configuration = try WatchBackgroundConfiguration(baseUrl: baseUrl, token: token, sessionId: sessionId, certificatePin: certificatePin)
    if relay.configuration != configuration { latest = "" }
    relay.configure(nil)
    do { try store.save(configuration) }
    catch { publishDisabled(); throw error }
    relay.configure(configuration); restoreError = nil
    if latest.isEmpty { publishBootstrap() }
  }
  func publish(_ payload: String) throws {
    start()
    guard payload.utf8.count <= 60_000, let configuration = relay.configuration,
      let bytes = payload.data(using: .utf8), let value = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
      value["session"] as? String == configuration.sessionId else {
      throw WatchBackgroundFailure(message: "Enable Apple Watch access for this paired Mac before syncing jobs.")
    }
    latest = payload
    if WCSession.isSupported(), WCSession.default.activationState == .activated {
      try WCSession.default.updateApplicationContext(["payload": payload])
    }
  }
  func disable() throws {
    start(); latest = ""; restoreError = nil; relay.configure(nil)
    defer { publishDisabled() }
    try store.clear()
  }
  private func publishBootstrap() {
    guard let configuration = relay.configuration else { return }
    let snapshot: [String: Any] = ["version": 1, "session": configuration.sessionId,
      "generatedAt": 0, "emittedAt": Date().timeIntervalSince1970 * 1_000,
      "connected": false, "serverName": "TensorFold", "jobs": [],
      "running": 0, "queued": 0, "approvals": 0, "transcriptReview": false,
      "error": "Refresh to connect to your Mac."]
    guard let bytes = try? JSONSerialization.data(withJSONObject: snapshot),
      let text = String(data: bytes, encoding: .utf8) else { return }
    try? publish(text)
  }
  private func publishDisabled() {
    if WCSession.isSupported(), WCSession.default.activationState == .activated {
      try? WCSession.default.updateApplicationContext(["disabled": true])
    }
  }
  private func publishResultSnapshot(_ result: [String: Any]) {
    guard let snapshot = result["snapshot"], let bytes = try? JSONSerialization.data(withJSONObject: snapshot),
      let text = String(data: bytes, encoding: .utf8) else { return }
    try? publish(text)
  }
  func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
    DispatchQueue.main.async { [weak self] in
      guard let self, activationState == .activated else { return }
      if self.relay.configuration != nil {
        if !self.latest.isEmpty { try? session.updateApplicationContext(["payload": self.latest]) }
      } else if self.restoreError == nil { self.publishDisabled() }
    }
  }
  func sessionDidBecomeInactive(_ session: WCSession) { }
  func sessionDidDeactivate(_ session: WCSession) { session.activate() }
  func session(_ session: WCSession, didReceiveMessage message: [String: Any], replyHandler: @escaping ([String: Any]) -> Void) {
    DispatchQueue.main.async { [weak self] in
      guard let self, let payload = message["payload"] as? String else { replyHandler(["error": "This watch request is invalid."]); return }
      if self.relay.configuration == nil { self.restoreConfiguration() }
      self.relay.handle(payload) { result in
        if let error = result["error"] as? String { replyHandler(["error": error]); return }
        guard let bytes = try? JSONSerialization.data(withJSONObject: result), bytes.count <= 60_000,
          let text = String(data: bytes, encoding: .utf8) else {
          replyHandler(["error": "The watch response was invalid; check the job before retrying."]); return
        }
        replyHandler(["payload": text])
      }
    }
  }
}
