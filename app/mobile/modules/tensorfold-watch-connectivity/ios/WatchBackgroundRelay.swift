import Foundation

protocol WatchBackgroundCancellable: AnyObject { func cancel() }
protocol WatchBackgroundBudget: AnyObject { func end() }

/// Main-thread state machine, independent of React and injectable for regression tests.
final class WatchBackgroundRelay {
  typealias Transport = (WatchBackgroundConfiguration, WatchBackgroundCommand, @escaping (Result<Data, Error>) -> Void) -> WatchBackgroundCancellable
  typealias Budget = (@escaping () -> Void) -> WatchBackgroundBudget?
  private final class Pending {
    let command: WatchBackgroundCommand
    let completion: ([String: Any]) -> Void
    var task: WatchBackgroundCancellable?
    var budget: WatchBackgroundBudget?
    init(_ command: WatchBackgroundCommand, _ completion: @escaping ([String: Any]) -> Void) {
      self.command = command; self.completion = completion
    }
  }
  private(set) var configuration: WatchBackgroundConfiguration?
  private var pending: [UUID: Pending] = [:]
  private let transport: Transport
  private let budget: Budget
  private let now: () -> TimeInterval
  var onResult: (([String: Any]) -> Void)?
  init(transport: @escaping Transport, budget: @escaping Budget, now: @escaping () -> TimeInterval = { Date().timeIntervalSince1970 }) {
    self.transport = transport; self.budget = budget; self.now = now
  }
  func configure(_ next: WatchBackgroundConfiguration?) {
    guard next != configuration else { return }
    configuration = next
    let entries = pending; pending.removeAll()
    for entry in entries.values {
      entry.task?.cancel(); entry.budget?.end()
      entry.completion(entry.command.failure("The phone connection changed; check the original job before retrying.", uncertain: true))
    }
  }
  func handle(_ payload: String, completion: @escaping ([String: Any]) -> Void) {
    guard let configuration else { completion(["error": "Enable Apple Watch access in TensorFold on your iPhone; unlock the phone once after restarting it."]); return }
    let command: WatchBackgroundCommand
    do { command = try WatchBackgroundCommand(payload: payload, session: configuration.sessionId, now: now()) }
    catch { completion(["error": "This watch request is invalid or stale; refresh before trying again."]); return }
    guard pending.count < 4 else { completion(command.failure("The phone is busy; wait and refresh the job.")); return }
    let identifier = UUID(), entry = Pending(command, completion)
    pending[identifier] = entry
    let executionBudget = budget { [weak self] in
      self?.finish(identifier, result: command.failure("iOS ended this request; check the job before retrying.", uncertain: true), cancel: true)
    }
    guard pending[identifier] != nil else { executionBudget?.end(); return }
    entry.budget = executionBudget
    guard executionBudget != nil else {
      finish(identifier, result: command.failure("iOS cannot run this request now; unlock your iPhone and try again."), cancel: true); return
    }
    let task = transport(configuration, command) { [weak self] response in
      guard let self, self.pending[identifier] != nil, self.configuration == configuration else { return }
      let result: [String: Any]
      do { result = try command.validateResponse(response.get(), session: configuration.sessionId) }
      catch {
        result = command.failure(command.mutatesReply ? "Delivery could not be confirmed; check the job before sending again." : "Could not reach your Mac securely; keep it awake and check its private host connection.", uncertain: true)
      }
      self.finish(identifier, result: result, cancel: false)
    }
    if pending[identifier] != nil { entry.task = task } else { task.cancel() }
  }
  private func finish(_ identifier: UUID, result: [String: Any], cancel: Bool) {
    guard let entry = pending.removeValue(forKey: identifier) else { return }
    if cancel { entry.task?.cancel() }
    entry.budget?.end()
    onResult?(result); entry.completion(result)
  }
}
