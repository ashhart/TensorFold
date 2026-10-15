import ExpoModulesCore
import Foundation
import Security
import CryptoKit

public final class TensorFoldPinnedNetworkModule: Module {
  public func definition() -> ModuleDefinition {
    Name("TensorFoldPinnedNetwork")
#if DEBUG
    AsyncFunction("testCertificate") { (base64: String, host: String, pin: String) -> [String: Any] in
      guard let bytes = Data(base64Encoded: base64), bytes.count <= 16_384,
        let certificate = SecCertificateCreateWithData(nil, bytes as CFData) else { return ["accepted": false, "reason": "Invalid fixture certificate"] }
      var trust: SecTrust?
      guard SecTrustCreateWithCertificates(certificate, SecPolicyCreateSSL(true, host as CFString), &trust) == errSecSuccess, let trust else { return ["accepted": false, "reason": "Invalid fixture trust"] }
      let failure = PinnedCertificate.failure(trust: trust, host: host, pin: pin)
      return ["accepted": failure == nil, "reason": failure ?? "Accepted exact pinned certificate"]
    }
#endif
    AsyncFunction("request") { (address: String, method: String, headers: [String: String], body: String?, pin: String, promise: Promise) in
      guard let url = URL(string: address), url.scheme == "https", let host = url.host,
        url.user == nil, url.password == nil, pin.range(of: "^[a-fA-F0-9]{64}$", options: .regularExpression) != nil,
        ["GET", "POST"].contains(method), (body?.utf8.count ?? 0) <= 65_536 else {
        promise.reject("E_PINNED_REQUEST", "Invalid private host request."); return
      }
      var request = URLRequest(url: url)
      request.httpMethod = method; request.httpBody = body?.data(using: .utf8); request.timeoutInterval = 15
      headers.forEach { request.setValue($0.value, forHTTPHeaderField: $0.key) }
      PinnedRequest(request: request, host: host, pin: pin.lowercased(), promise: promise).start()
    }
  }
}

private final class PinnedRequest: NSObject, URLSessionDataDelegate, URLSessionTaskDelegate, @unchecked Sendable {
  private let request: URLRequest
  private let host: String
  private let pin: String
  private let promise: Promise
  private var response: HTTPURLResponse?
  private var data = Data()
  private var failure: String?
  private let limit = 8 * 1024 * 1024
  private lazy var session: URLSession = {
    let config = URLSessionConfiguration.ephemeral
    config.httpCookieStorage = nil; config.urlCredentialStorage = nil; config.urlCache = nil
    config.timeoutIntervalForRequest = 15; config.timeoutIntervalForResource = 15
    return URLSession(configuration: config, delegate: self, delegateQueue: nil)
  }()
  init(request: URLRequest, host: String, pin: String, promise: Promise) {
    self.request = request; self.host = host; self.pin = pin; self.promise = promise
  }
  func start() { session.dataTask(with: request).resume() }
  func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge, completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
    guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
      challenge.protectionSpace.host == host, let trust = challenge.protectionSpace.serverTrust else {
      failure = "The private host certificate could not be verified."; completionHandler(.cancelAuthenticationChallenge, nil); return
    }
    if let reason = PinnedCertificate.failure(trust: trust, host: host, pin: pin) { failure = reason; completionHandler(.cancelAuthenticationChallenge, nil); return }
    completionHandler(.useCredential, URLCredential(trust: trust))
  }
  func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
    failure = "Redirects are not allowed for a paired private host."; completionHandler(nil)
  }
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse, completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
    guard let http = response as? HTTPURLResponse, response.expectedContentLength <= Int64(limit) else {
      failure = "The private host returned an unsupported or oversized response."; completionHandler(.cancel); return
    }
    self.response = http; completionHandler(.allow)
  }
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive chunk: Data) {
    if data.count + chunk.count > limit { failure = "The private host response is too large."; dataTask.cancel(); return }
    data.append(chunk)
  }
  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    defer { session.invalidateAndCancel() }
    if let failure { promise.reject("E_PINNED_HOST", failure); return }
    if error != nil { promise.reject("E_PINNED_HOST", "Could not reach the pinned private host."); return }
    guard let response, let body = String(data: data, encoding: .utf8) else { promise.reject("E_PINNED_HOST", "The private host returned invalid data."); return }
    var headers: [String: String] = [:]
    response.allHeaderFields.forEach { headers[String(describing: $0.key)] = String(describing: $0.value) }
    promise.resolve(["status": response.statusCode, "headers": headers, "body": body])
  }
}
