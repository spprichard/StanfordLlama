// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "StanfordLlama",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "StanfordLlama",
            targets: ["StanfordLlama"]),
        .executable(
            name: "Server",
            targets: [
                "Server"
            ]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/StanfordBDHG/llama.cpp", .upToNextMinor(from: "0.2.1")),
        .package(url: "https://github.com/vapor/vapor.git", from: "4.94.1"),
    ],
    targets: [
        .executableTarget(
            name: "Server",
            dependencies: [
                .product(name: "Vapor", package: "vapor"),
                "StanfordLlama"
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),
        .target(
            name: "StanfordLlama",
            dependencies: [
                .product(name: "llama", package: "llama.cpp")
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .enableExperimentalFeature("StrictConcurrency"),
            ]
        ),
        .testTarget(
            name: "StanfordLlamaTests",
            dependencies: ["StanfordLlama"]),
    ]
)
