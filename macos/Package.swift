// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HoopVision",
    platforms: [
        .macOS(.v15)
    ],
    targets: [
        .executableTarget(
            name: "HoopVision",
            path: "Sources"
        )
    ]
)
