import Foundation
import AppKit

class PythonBridge: ObservableObject, @unchecked Sendable {
    @Published var isRunning = false
    @Published var progress: Double = 0
    @Published var currentStep: String = ""
    @Published var eventsFound: Int = 0
    @Published var framesProcessed: Int = 0
    @Published var error: String?
    @Published var allOutput: String = ""

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

        var env = ProcessInfo.processInfo.environment
        let existingPythonPath = env["PYTHONPATH"] ?? ""
        env["PYTHONPATH"] = existingPythonPath.isEmpty ? projectRoot : "\(projectRoot):\(existingPythonPath)"
        env["HOME"] = NSHomeDirectory()
        env["USER"] = NSUserName()
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\(env["PATH"] ?? "")"
        process?.environment = env

        outputPipe = Pipe()
        errorPipe = Pipe()
        process?.standardOutput = outputPipe
        process?.standardError = errorPipe

        let output = OutputCollector()

        outputPipe?.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            if let str = String(data: data, encoding: .utf8) {
                output.addStdout(str)
                DispatchQueue.main.async {
                    self.parseProgressOutput(str)
                    self.allOutput = output.combined()
                }
            }
        }

        errorPipe?.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            if let str = String(data: data, encoding: .utf8) {
                output.addStderr(str)
                DispatchQueue.main.async {
                    self.allOutput = output.combined()
                }
            }
        }

        process?.terminationHandler = { proc in
            DispatchQueue.main.async {
                self.isRunning = false
                self.progress = 1.0
                self.currentStep = "Complete"
                let combined = output.combined()
                self.allOutput = "STDOUT:\n" + output.stdout + "\n\nSTDERR:\n" + output.stderr
                if proc.terminationStatus != 0 {
                    self.error = output.stderr.isEmpty ? "Process exited with code \(proc.terminationStatus)" : output.stderr
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
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            currentStep = trimmed
        }

        if text.contains("Frames analyzed") {
            if let match = text.range(of: #"\d+"#, options: .regularExpression) {
                framesProcessed = Int(text[match]) ?? framesProcessed
            }
            progress = max(progress, 0.95)
        }
        if text.contains("Events detected") {
            if let match = text.range(of: #"\d+"#, options: .regularExpression) {
                eventsFound = Int(text[match]) ?? eventsFound
            }
        }
        if text.contains("Frame size") {
            progress = 0.15
        }
        if text.contains("Analyzing") {
            progress = 0.30
        }
        if text.contains("Progress:") {
            if let match = text.range(of: #"\d+"#, options: .regularExpression) {
                let f = Int(text[match]) ?? 0
                framesProcessed = f
                progress = 0.30 + Double(f) * 0.01
            }
        }
        if text.contains("Done!") {
            progress = 1.0
        }
        if text.contains("Error:") || text.contains("error") {
            // Don't overwrite error, just track it
        }
    }

    func cancel() {
        process?.terminate()
        isRunning = false
    }
}

// MARK: - Output Collector (thread-safe)

final class OutputCollector: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var stdout = ""
    private(set) var stderr = ""

    func addStdout(_ s: String) {
        lock.lock()
        stdout += s
        lock.unlock()
    }

    func addStderr(_ s: String) {
        lock.lock()
        stderr += s
        lock.unlock()
    }

    func combined() -> String {
        lock.lock()
        defer { lock.unlock() }
        return stdout + stderr
    }
}
