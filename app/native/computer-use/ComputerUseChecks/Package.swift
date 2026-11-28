// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ComputerUseChecks",
    targets: [
        .target(name: "ComputerUseChecks"),
        .testTarget(name: "ComputerUseChecksTests", dependencies: ["ComputerUseChecks"]),
    ]
)
