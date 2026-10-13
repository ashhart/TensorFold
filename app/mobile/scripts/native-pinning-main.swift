import Foundation
import Security

struct Fixture: Decodable { let name: String; let host: String; let certificate: String; let pin: String; let expected: Bool }
let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
var failed = 0
for fixture in fixtures {
  guard let data = Data(base64Encoded: fixture.certificate), let certificate = SecCertificateCreateWithData(nil, data as CFData) else { fatalError("Invalid public fixture") }
  var trust: SecTrust?
  precondition(SecTrustCreateWithCertificates(certificate, SecPolicyCreateSSL(true, fixture.host as CFString), &trust) == errSecSuccess)
  let reason = PinnedCertificate.failure(trust: trust!, host: fixture.host, pin: fixture.pin)
  let passed = (reason == nil) == fixture.expected
  if !passed { failed += 1 }
  print("\(passed ? "PASS" : "FAIL"): \(fixture.name) \(reason ?? "accepted")")
}
exit(failed == 0 ? 0 : 1)
