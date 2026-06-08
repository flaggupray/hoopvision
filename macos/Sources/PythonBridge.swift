import Foundation
import AppKit

class PythonBridge: ObservableObject, @unchecked Sendable {
    @Published var isRunning = false
    @Published var progress: Double = 0
    @Published var currentStep: String = ""
    @Published var eventsFound: Int = 0
    @Published var framesProcessed: Int = 0
    @Published var error: String?

    private var process: Process?
    private var outputPipe: Pipe?
    private var errorPipe: Pipe?

    func findEnginePath() -> String {
        // Look relative to the app bundle first, then fall back to hardcoded path
        if let bundlePath = Bundle.main.resourcePath {
            let relativePath = (bundlePath as NSString)
                .appendingPathComponent("../../../../engine")
            let standardized = (relativePath as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: standardized) {
                return standardized
            }
        }
        // Default path for development
        let devPath = "/Users/mac/hoopvision/engine"
        if FileManager.default.fileExists(atPath: devPath) {
            return devPath
        }
        return ""
    }

    func findProjectRoot() -> String {
        if let bundlePath = Bundle.main.resourcePath {
            let relativePath = (bundlePath as NSString)
                .appendingPathComponent("../../../../")
            let standardized = (relativePath as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: standardized + "/engine/cli.py") {
                return standardized
            }
        }
        return "/Users/mac/hoopvision"
    }

    func runAnalysis(
        videoPath: String,
        outputPath: String,
        device: String,
        sampleRate: Int,
        enableOCR: Bool,
        enableClassifier: Bool,
        confidence: Double,
        homeTeam: String,
        awayTeam: String,
        gameDate: String,
        format: String = "html"
    ) {
        isRunning = true
        progress = 0
        currentStep = "Starting..."
        eventsFound = 0
        framesProcessed = 0
        error = nil

        let projectRoot = findProjectRoot()

        var args: [String] = [
            "-m", "engine.cli", "run",
            videoPath,
            "--output", outputPath,
            "--format", format,
            "--device", device,
            "--sample-rate", "\(sampleRate)",
            "--confidence", "\(confidence)",
            "--home", homeTeam.isEmpty ? "Home" : homeTeam,
            "--away", awayTeam.isEmpty ? "Away" : awayTeam,
        ]
        if !gameDate.isEmpty {
            args.append(contentsOf: ["--date", gameDate])
        }
        if !enableOCR {
            args.append("--no-ocr")
        }
        if !enableClassifier {
            args.append("--no-classify")
        }

        process = Process()
        process?.executableURL = URL(fileURLWithPath: "/usr/local/bin/python3")
        process?.arguments = args
        process?.currentDirectoryURL = URL(fileURLWithPath: projectRoot)

        outputPipe = Pipe()
        errorPipe = Pipe()
        process?.standardOutput = outputPipe
        process?.standardError = errorPipe

        outputPipe?.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self = self else { return }
            let data = handle.availableData
            guard !data.isEmpty else { return }
            if let str = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    self.parseProgressOutput(str)
                }
            }
        }

        process?.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                self?.isRunning = false
                self?.progress = 1.0
                self?.currentStep = "Complete"
                if proc.terminationStatus != 0 {
                    let errData = self?.errorPipe?.fileHandleForReading.readDataToEndOfFile() ?? Data()
                    let errStr = String(data: errData, encoding: .utf8) ?? "Unknown error"
                    self?.error = errStr
                }
            }
        }

        do {
            try process?.run()
        } catch {
            isRunning = false
            self.error = error.localizedDescription
        }
    }

    private func parseProgressOutput(_ text: String) {
        currentStep = text.trimmingCharacters(in: .whitespacesAndNewlines)

        if text.contains("Frames analyzed") {
            if let match = text.range(of: #"\d+"#, options: .regularExpression) {
                framesProcessed = Int(text[match]) ?? framesProcessed
            }
            progress = min(progress + 0.15, 0.95)
        }
        if text.contains("Events detected") {
            if let match = text.range(of: #"\d+"#, options: .regularExpression) {
                eventsFound = Int(text[match]) ?? eventsFound
            }
            progress = min(progress + 0.2, 0.95)
        }
        if text.contains("Analyzing") {
            progress = 0.3
        }
        if text.contains("Done!") {
            progress = 1.0
        }
    }

    func cancel() {
        process?.terminate()
        isRunning = false
    }
}
