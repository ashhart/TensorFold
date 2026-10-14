import Foundation
import Security
import CryptoKit

enum PinnedCertificate {
  static func failure(trust: SecTrust, host: String, pin: String) -> String? {
    guard let certificates = SecTrustCopyCertificateChain(trust) as? [SecCertificate], let certificate = certificates.first else { return "The private host supplied no certificate." }
    let digest = SHA256.hash(data: SecCertificateCopyData(certificate) as Data).map { String(format: "%02x", $0) }.joined()
    guard digest == pin.lowercased() else { return "The host certificate changed; verify it on your Mac and pair again." }
    SecTrustSetPolicies(trust, SecPolicyCreateSSL(true, host as CFString))
    SecTrustSetAnchorCertificates(trust, [certificate] as CFArray)
    SecTrustSetAnchorCertificatesOnly(trust, true)
    return SecTrustEvaluateWithError(trust, nil) ? nil : "The pinned certificate is expired or does not match the private host address."
  }
}
