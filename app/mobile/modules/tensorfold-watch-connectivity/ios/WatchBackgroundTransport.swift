import Foundation
import Security
#if canImport(UIKit)
import UIKit
#endif
#if canImport(TensorFoldPinnedNetwork)
import TensorFoldPinnedNetwork
#endif

/// A short foreground URLSession request is permitted while Watch Connectivity
/// wakes iOS; this does not claim continuous background execution.
final class WatchBackgroundRequest: NSObject, URLSessionDataDelegate, URLSessionTaskDelegate, WatchBackgroundCancellable, @unchecked Sendable {
  private let configuration: WatchBackgroundConfiguration
  private let command: WatchBackgroundCommand
  private let completion: (Result<Data, Error>) -> Void
  private var bytes = Data()
  private var accepted = false
  private var failure: String?
  private var completed = false
  private var task: URLSessionDataTask?
  private lazy var session: URLSession = {
    let config = URLSessionConfiguration.ephemeral
    config.httpCookieStorage = nil; config.httpShouldSetCookies = false
    config.urlCredentialStorage = nil; config.urlCache = nil
    config.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
    config.timeoutIntervalForRequest = 15; config.timeoutIntervalForResource = 15
    config.waitsForConnectivity = false
    return URLSession(configuration: config, delegate: self, delegateQueue: nil)
  }()
  init(configuration: WatchBackgroundConfiguration, command: WatchBackgroundCommand, completion: @escaping (Result<Data, Error>) -> Void) {
    self.configuration = configuration; self.command = command; self.completion = completion
  }
  func start() {
    var request = URLRequest(url: configuration.endpoint, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 15)
    request.httpMethod = "POST"; request.httpBody = command.payload; request.httpShouldHandleCookies = false
    request.setValue("Bearer " + configuration.token, forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    task = session.dataTask(with: request); task?.resume()
  }
  func cancel() { task?.cancel(); session.invalidateAndCancel() }
  func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge, completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
    guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
      challenge.protectionSpace.host == configuration.endpoint.host, let trust = challenge.protectionSpace.serverTrust,
      PinnedCertificate.failure(trust: trust, host: configuration.endpoint.host!, pin: configuration.certificatePin) == nil else {
      failure = "The Mac certificate could not be verified; pair again on your iPhone."
      completionHandler(.cancelAuthenticationChallenge, nil); return
    }
    completionHandler(.useCredential, URLCredential(trust: trust))
  }
  func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
    failure = "The Mac tried to redirect the paired connection."; completionHandler(nil)
  }
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse, completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
    guard let response = response as? HTTPURLResponse, response.url == configuration.endpoint,
      (200...299).contains(response.statusCode), response.expectedContentLength <= 60_000,
      response.mimeType?.lowercased() == "application/json" else {
      failure = "The paired Mac returned an unsupported response."; completionHandler(.cancel); return
    }
    accepted = true; completionHandler(.allow)
  }
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    guard bytes.count + data.count <= 60_000 else { failure = "The Mac response is too large."; dataTask.cancel(); return }
    bytes.append(data)
  }
  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    guard !completed else { return }; completed = true
    let result: Result<Data, Error> = failure == nil && error == nil && accepted
      ? .success(bytes) : .failure(WatchBackgroundFailure(message: failure ?? "The paired Mac did not respond in time."))
    session.finishTasksAndInvalidate()
    DispatchQueue.main.async { self.completion(result) }
  }
}

#if canImport(UIKit)
final class WatchBackgroundExecutionBudget: WatchBackgroundBudget {
  private var identifier: UIBackgroundTaskIdentifier = .invalid
  private var timer: Timer?
  init?(expiration: @escaping () -> Void) {
    identifier = UIApplication.shared.beginBackgroundTask(withName: "TensorFold watch request") { expiration() }
    guard identifier != .invalid else { return nil }
    timer = Timer.scheduledTimer(withTimeInterval: 16, repeats: false) { _ in expiration() }
  }
  func end() {
    timer?.invalidate(); timer = nil
    guard identifier != .invalid else { return }
    UIApplication.shared.endBackgroundTask(identifier); identifier = .invalid
  }
  deinit { end() }
}

#endif
