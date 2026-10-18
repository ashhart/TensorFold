import ExpoModulesCore
import UIKit

public final class TensorFoldWatchAppDelegateSubscriber: ExpoAppDelegateSubscriber {
  public func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
    WatchPhoneSession.shared.start()
    return true
  }
}
