import Foundation
import CoreFoundation

struct WatchBackgroundFailure: Error, LocalizedError {
  let message: String
  var errorDescription: String? { message }
}

/// Phone-only configuration: never included in Watch Connectivity payloads.
struct WatchBackgroundConfiguration: Codable, Equatable {
  let baseUrl: String
  let token: String
  let sessionId: String
  let certificatePin: String

  init(baseUrl: String, token: String, sessionId: String, certificatePin: String) throws {
    guard baseUrl.utf8.count <= 2_048, let parts = URLComponents(string: baseUrl),
      parts.scheme == "https", let host = parts.host, !host.isEmpty,
      parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
      parts.port == nil || (1...65_535).contains(parts.port!),
      !parts.path.contains("\\"), !parts.path.split(separator: "/").contains(where: { $0 == "." || $0 == ".." }),
      token.utf8.count >= 16, token.utf8.count <= 512, token.utf8.allSatisfy({ $0 >= 33 && $0 <= 126 }),
      sessionId.range(of: "^[a-zA-Z0-9-]{8,80}$", options: .regularExpression) != nil,
      certificatePin.range(of: "^[a-fA-F0-9]{64}$", options: .regularExpression) != nil else {
      throw WatchBackgroundFailure(message: "Pair with your private HTTPS Mac host and its certificate before enabling background Apple Watch access.")
    }
    self.baseUrl = baseUrl.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    self.token = token; self.sessionId = sessionId; self.certificatePin = certificatePin.lowercased()
  }
  var endpoint: URL { URL(string: baseUrl + "/v1/watch/command")! }
  func validated() throws -> WatchBackgroundConfiguration {
    try WatchBackgroundConfiguration(baseUrl: baseUrl, token: token, sessionId: sessionId, certificatePin: certificatePin)
  }
}

struct WatchBackgroundCommand {
  let id: String
  let action: String
  let payload: Data
  var mutatesReply: Bool { action == "reply" }

  init(payload: String, session: String, now: TimeInterval) throws {
    let invalid = WatchBackgroundFailure(message: "This watch request is invalid or stale; refresh before trying again.")
    guard payload.utf8.count <= 12_000, let data = payload.data(using: .utf8),
      let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(value.keys).isSubset(of: ["version", "id", "action", "session", "threadId", "text", "itemId", "speechId", "sequence", "generatedAt"]),
      let version = value["version"] as? NSNumber, CFGetTypeID(version) != CFBooleanGetTypeID(), version.intValue == 1, version.doubleValue == 1,
      let id = value["id"] as? String, id.range(of: "^[a-zA-Z0-9-]{8,80}$", options: .regularExpression) != nil,
      value["session"] as? String == session, let action = value["action"] as? String,
      ["refresh", "reply", "audio-start", "audio-next", "audio-stop"].contains(action),
      let generatedAt = value["generatedAt"] as? NSNumber, CFGetTypeID(generatedAt) != CFBooleanGetTypeID(),
      generatedAt.doubleValue.isFinite, abs(now * 1_000 - generatedAt.doubleValue) <= 45_000 else { throw invalid }
    func bounded(_ name: String, _ maximum: Int) -> Bool {
      guard let text = value[name] as? String else { return false }
      return !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && text.utf16.count <= maximum
    }
    if action == "reply", !bounded("threadId", 256) || !bounded("text", 2_000) { throw invalid }
    if action == "audio-start", !bounded("threadId", 256) || !bounded("itemId", 256) { throw invalid }
    if action == "audio-next" || action == "audio-stop" {
      guard let speechId = value["speechId"] as? String, speechId.range(of: "^[a-zA-Z0-9-]{8,80}$", options: .regularExpression) != nil else { throw invalid }
    }
    if action == "audio-next" {
      guard let sequence = value["sequence"] as? NSNumber, CFGetTypeID(sequence) != CFBooleanGetTypeID(),
        sequence.doubleValue == Double(sequence.intValue), (0...10_000).contains(sequence.intValue) else { throw invalid }
    }
    self.id = id; self.action = action; self.payload = data
  }
  func failure(_ message: String, uncertain: Bool = false) -> [String: Any] {
    var value: [String: Any] = ["id": id, "ok": false, "message": message]
    if uncertain && mutatesReply { value["uncertain"] = true }
    return value
  }
  func validateResponse(_ data: Data, session: String) throws -> [String: Any] {
    guard data.count <= 60_000, let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      result["id"] as? String == id, let success = result["ok"] as? NSNumber, CFGetTypeID(success) == CFBooleanGetTypeID(), let message = result["message"] as? String, message.utf16.count <= 500,
      Set(result.keys).isSubset(of: ["id", "ok", "message", "uncertain", "snapshot", "speech", "audio"]),
      result["uncertain"] == nil || (result["uncertain"] as? NSNumber).map({ CFGetTypeID($0) == CFBooleanGetTypeID() }) == true else {
      throw WatchBackgroundFailure(message: "The Mac returned an invalid watch response.")
    }
    if let snapshot = result["snapshot"] {
      guard let object = snapshot as? [String: Any], object["session"] as? String == session else {
        throw WatchBackgroundFailure(message: "The Mac connection changed; refresh before trying again.")
      }
    }
    return result
  }
}
