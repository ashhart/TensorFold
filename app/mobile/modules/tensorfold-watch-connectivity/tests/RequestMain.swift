import Foundation

@main struct NativeRequestFixture {
  static func main() throws {
    let config = try WatchBackgroundConfiguration(baseUrl: CommandLine.arguments[1], token: "fixture-phone-only-token", sessionId: "session-test-1", certificatePin: CommandLine.arguments[2])
    let command = try WatchBackgroundCommand(payload: "{\"id\":\"fixture-command-1\",\"version\":1,\"session\":\"session-test-1\",\"action\":\"refresh\",\"generatedAt\":1000000}", session: config.sessionId, now: 1_000)
    var done = false, success = false, count = 0
    let request = WatchBackgroundRequest(configuration: config, command: command) { result in
      if case let .success(data) = result { success = true; count = data.count }
      done = true
    }
    request.start()
    let deadline = Date().addingTimeInterval(18)
    while !done && Date() < deadline { RunLoop.current.run(until: Date().addingTimeInterval(0.01)) }
    if !done { request.cancel() }
    print("{\"completed\":\(done),\"accepted\":\(success),\"bytes\":\(count)}")
  }
}
