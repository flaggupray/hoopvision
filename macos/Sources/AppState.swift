import SwiftUI

enum AppScreen {
    case welcome
    case importVideo
    case configure
    case analyzing
    case result
}

class AppState: ObservableObject {
    @Published var screen: AppScreen = .welcome
    @Published var enginePath: String = ""
    @Published var defaultDevice: String = "auto"
    @Published var sampleRate: Int = 6
    @Published var enableOCR: Bool = true
    @Published var enableClassifier: Bool = true
    @Published var confidenceThreshold: Double = 0.5
    @Published var homeTeam: String = ""
    @Published var awayTeam: String = ""
    @Published var gameDate: String = ""

    // Analysis state
    @Published var analysisProgress: Double = 0
    @Published var analysisEventsFound: Int = 0
    @Published var analysisFramesProcessed: Int = 0
    @Published var analysisStatusText: String = ""
    @Published var analysisError: String?

    // Result
    @Published var outputPath: String = ""
    @Published var selectedVideoPath: String = ""

    func reset() {
        screen = .welcome
        analysisProgress = 0
        analysisEventsFound = 0
        analysisFramesProcessed = 0
        analysisStatusText = ""
        analysisError = nil
        outputPath = ""
        selectedVideoPath = ""
    }

    func startNewAnalysis() {
        reset()
        screen = .importVideo
    }
}
