// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ComputerUseChecks",
    targets: [
        .target(
            name: "ComputerUseChecks",
            path: "ComputerUseChecks",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "ComputerUseChecksTests",
            dependencies: ["ComputerUseChecks"],
            path: "Tests/ComputerUseChecksTests",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
