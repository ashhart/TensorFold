import Foundation
import Security

protocol WatchBackgroundKeychain {
  func delete(_ query: [String: Any]) -> OSStatus
  func read(_ query: [String: Any]) -> (OSStatus, Data?)
  func update(_ query: [String: Any], _ attributes: [String: Any]) -> OSStatus
  func add(_ attributes: [String: Any]) -> OSStatus
}
private struct SystemWatchBackgroundKeychain: WatchBackgroundKeychain {
  func delete(_ query: [String: Any]) -> OSStatus { SecItemDelete(query as CFDictionary) }
  func read(_ query: [String: Any]) -> (OSStatus, Data?) {
    var result: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &result)
    return (status, result as? Data)
  }
  func update(_ query: [String: Any], _ attributes: [String: Any]) -> OSStatus { SecItemUpdate(query as CFDictionary, attributes as CFDictionary) }
  func add(_ attributes: [String: Any]) -> OSStatus { SecItemAdd(attributes as CFDictionary, nil) }
}

/// Explicit watch consent gets its own device-only Keychain item; the regular
/// phone session retains its stricter WHEN_UNLOCKED accessibility.
final class WatchBackgroundConfigurationStore {
  private let defaults: UserDefaults
  private let service: String
  private let keychain: WatchBackgroundKeychain
  private let installKey = "tensorfold.watch.background.install.v1"
  private let consentKey = "tensorfold.watch.background.consent.v1"
  init(defaults: UserDefaults = .standard, service: String = (Bundle.main.bundleIdentifier ?? "dev.tensorfold.mobile") + ".watch-background.v1", keychain: WatchBackgroundKeychain? = nil) {
    self.defaults = defaults; self.service = service; self.keychain = keychain ?? SystemWatchBackgroundKeychain()
  }
  private var query: [String: Any] {
    [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
     kSecAttrAccount as String: "paired-watch-host", kSecAttrSynchronizable as String: false]
  }
  private func prepareInstallation() throws {
    guard !defaults.bool(forKey: installKey) else { return }
    // Keychain items survive reinstall, app preferences do not. Never resurrect
    // a previous installation's consent (including a deferred first-unlock).
    defaults.set(false, forKey: consentKey)
    let status = keychain.delete(query)
    guard status == errSecSuccess || status == errSecItemNotFound else {
      throw WatchBackgroundFailure(message: "Unlock your iPhone once before enabling Apple Watch access.")
    }
    defaults.set(true, forKey: installKey)
  }
  func load() throws -> WatchBackgroundConfiguration? {
    try prepareInstallation()
    guard defaults.bool(forKey: consentKey) else { return nil }
    var attributes = query
    attributes[kSecReturnData as String] = true; attributes[kSecMatchLimit as String] = kSecMatchLimitOne
    let (status, result) = keychain.read(attributes)
    if status == errSecItemNotFound { defaults.set(false, forKey: consentKey); return nil }
    guard status == errSecSuccess, let data = result, data.count <= 4_096 else {
      throw WatchBackgroundFailure(message: "Unlock your iPhone once after restarting it to reconnect your watch.")
    }
    return try JSONDecoder().decode(WatchBackgroundConfiguration.self, from: data).validated()
  }
  func save(_ configuration: WatchBackgroundConfiguration) throws {
    try prepareInstallation()
    let validated = try configuration.validated(), data = try JSONEncoder().encode(validated)
    guard data.count <= 4_096 else { throw WatchBackgroundFailure(message: "This private host configuration is too large.") }
    defaults.set(false, forKey: consentKey)
    let changes: [String: Any] = [kSecValueData as String: data, kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly]
    var status = keychain.update(query, changes)
    if status == errSecItemNotFound {
      var attributes = query; changes.forEach { attributes[$0.key] = $0.value }
      status = keychain.add(attributes)
    }
    guard status == errSecSuccess else {
      throw WatchBackgroundFailure(message: "Background Apple Watch access could not be saved securely; unlock your iPhone and enable it again.")
    }
    defaults.set(true, forKey: consentKey)
  }
  func clear() throws {
    // The app-local gate remains revoked even if Keychain deletion is deferred.
    defaults.set(false, forKey: consentKey)
    let status = keychain.delete(query)
    guard status == errSecSuccess || status == errSecItemNotFound else {
      throw WatchBackgroundFailure(message: "Apple Watch access is disabled; unlock your iPhone to finish removing its background credential.")
    }
  }
}
