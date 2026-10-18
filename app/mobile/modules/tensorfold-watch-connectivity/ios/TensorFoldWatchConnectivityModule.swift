import ExpoModulesCore
import Foundation

public final class TensorFoldWatchConnectivityModule: Module {
  public func definition() -> ModuleDefinition {
    Name("TensorFoldWatchConnectivity")
    AsyncFunction("configureBackground") { (input: [String: String]) in
      guard Set(input.keys) == ["baseUrl", "token", "sessionId", "certificatePin"],
        let baseUrl = input["baseUrl"], let token = input["token"], let sessionId = input["sessionId"], let pin = input["certificatePin"] else {
        throw WatchBackgroundFailure(message: "Pair with your Mac before enabling Apple Watch access.")
      }
      try WatchPhoneSession.shared.configureBackground(baseUrl: baseUrl, token: token, sessionId: sessionId, certificatePin: pin)
    }.runOnQueue(.main)
    AsyncFunction("backgroundStatus") { WatchPhoneSession.shared.backgroundStatus() }.runOnQueue(.main)
    AsyncFunction("publish") { (payload: String) in try WatchPhoneSession.shared.publish(payload) }.runOnQueue(.main)
    AsyncFunction("disable") { try WatchPhoneSession.shared.disable() }.runOnQueue(.main)
    // Consent is revoked only by an explicit disable/forget/account switch,
    // never by React OnDestroy or a foreground/background lifecycle transition.
  }
}
