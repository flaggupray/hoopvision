import SwiftUI

struct AnalysisProgressView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var bridge = PythonBridge()

    @State private var progressScale: CGFloat = 1.0

    var body: some View {
        VStack(spacing: 36) {
            Spacer()

            // Animated progress ring
            ZStack {
                Circle()
                    .stroke(Color.black.opacity(0.06), lineWidth: 8)
                    .frame(width: 160, height: 160)

                Circle()
                    .trim(from: 0, to: bridge.progress)
                    .stroke(
                        LinearGradient(
                            colors: [.orange, Color.orange.opacity(0.7)],
                            startPoint: .topLeading, endPoint: .bottomTrailing
                        ),
                        style: StrokeStyle(lineWidth: 8, lineCap: .round)
                    )
                    .frame(width: 160, height: 160)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.5), value: bridge.progress)

                VStack(spacing: 4) {
                    if bridge.error != nil {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 32))
                            .foregroundStyle(.red)
                    } else if bridge.isRunning {
                        Image(systemName: "basketball.fill")
                            .font(.system(size: 32))
                            .foregroundStyle(.orange)
                            .scaleEffect(progressScale)
                            .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: progressScale)
                    } else {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 36))
                            .foregroundStyle(.green)
                    }

                    Text("\(Int(bridge.progress * 100))%")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                }
            }
            .onAppear {
                progressScale = 1.3
                startBridgeAnalysis()
            }

            // Status text
            VStack(spacing: 8) {
                Text(bridge.currentStep)
                    .font(.body)
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 320)

                if bridge.framesProcessed > 0 {
                    HStack(spacing: 20) {
                        StatLabel(label: "Frames", value: "\(bridge.framesProcessed)")
                        StatLabel(label: "Events", value: "\(bridge.eventsFound)")
                    }
                }

                if let error = bridge.error {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 360)
                        .padding()
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
            }

            Spacer()

            // Cancel / Done buttons
            HStack(spacing: 16) {
                if bridge.isRunning {
                    GlassButton(title: "Cancel", icon: "stop.fill") {
                        bridge.cancel()
                    }
                } else {
                    GlassButton(title: "Back", icon: "chevron.left") {
                        appState.screen = .configure
                    }

                    if bridge.error == nil {
                        GlassButton(title: "View Timeline", icon: "list.clipboard", isPrimary: true) {
                            appState.screen = .result
                        }
                    }
                }
            }
            .padding(.bottom, 48)
        }
        .padding(48)
    }

    private func startBridgeAnalysis() {
        let outputPath = appState.outputPath.isEmpty
            ? NSTemporaryDirectory() + "timeline.html"
            : appState.outputPath

        bridge.runAnalysis(
            videoPath: appState.selectedVideoPath,
            outputPath: outputPath,
            device: appState.defaultDevice,
            sampleRate: appState.sampleRate,
            enableOCR: appState.enableOCR,
            enableClassifier: appState.enableClassifier,
            confidence: appState.confidenceThreshold,
            homeTeam: appState.homeTeam,
            awayTeam: appState.awayTeam,
            gameDate: appState.gameDate
        )
    }
}

struct StatLabel: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 3) {
            Text(value)
                .font(.system(.title3, design: .rounded).bold())
                .foregroundStyle(.primary)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
        }
    }
}
