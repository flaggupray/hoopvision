import SwiftUI

struct ConfigView: View {
    @EnvironmentObject var appState: AppState
    @State private var outputFileName = "timeline"

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 8) {
                Text("Analysis Configuration")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                Text("Configure how the AI analyzes your game")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 40)

            ScrollView {
                VStack(spacing: 24) {
                    // Video info
                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            Label("Video", systemImage: "film")
                                .font(.headline)
                            Text((appState.selectedVideoPath as NSString).lastPathComponent)
                                .font(.callout)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                    }
                    .glassCard(padding: 0, cornerRadius: 18)

                    // Game info
                    GroupBox {
                        VStack(spacing: 14) {
                            Label("Game Info", systemImage: "sportscourt")
                                .font(.headline)
                                .frame(maxWidth: .infinity, alignment: .leading)

                            HStack(spacing: 16) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Home Team").font(.caption).foregroundStyle(.secondary)
                                    TextField("e.g. Lakers", text: $appState.homeTeam)
                                        .textFieldStyle(.roundedBorder)
                                }
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Away Team").font(.caption).foregroundStyle(.secondary)
                                    TextField("e.g. Warriors", text: $appState.awayTeam)
                                        .textFieldStyle(.roundedBorder)
                                }
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Date").font(.caption).foregroundStyle(.secondary)
                                    TextField("e.g. 2024-12-25", text: $appState.gameDate)
                                        .textFieldStyle(.roundedBorder)
                                }
                            }
                        }
                        .padding(8)
                    }
                    .glassCard(padding: 0, cornerRadius: 18)

                    // Engine settings
                    GroupBox {
                        VStack(spacing: 14) {
                            Label("Engine Settings", systemImage: "gearshape.2")
                                .font(.headline)
                                .frame(maxWidth: .infinity, alignment: .leading)

                            HStack(spacing: 20) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Inference Device").font(.caption).foregroundStyle(.secondary)
                                    Picker("Device", selection: $appState.defaultDevice) {
                                        Text("Auto").tag("auto")
                                        Text("CPU").tag("cpu")
                                        Text("GPU (MPS)").tag("mps")
                                    }
                                    .pickerStyle(.segmented)
                                    .frame(width: 260)
                                }

                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Sample Rate (fps)").font(.caption).foregroundStyle(.secondary)
                                    Picker("Sample Rate", selection: $appState.sampleRate) {
                                        Text("2 fps").tag(2)
                                        Text("4 fps").tag(4)
                                        Text("6 fps").tag(6)
                                        Text("10 fps").tag(10)
                                    }
                                    .pickerStyle(.segmented)
                                    .frame(width: 220)
                                }
                            }

                            HStack(spacing: 20) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Confidence Threshold").font(.caption).foregroundStyle(.secondary)
                                    Slider(value: $appState.confidenceThreshold, in: 0.1...0.9, step: 0.05) {
                                        Text("Confidence")
                                    }
                                    Text("\(appState.confidenceThreshold, format: .number.precision(.fractionLength(2)))")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                .frame(maxWidth: 240)
                            }

                            HStack(spacing: 24) {
                                Toggle("Jersey Number OCR", isOn: $appState.enableOCR)
                                    .toggleStyle(.switch)
                                Toggle("Action Classification", isOn: $appState.enableClassifier)
                                    .toggleStyle(.switch)
                            }
                        }
                        .padding(8)
                    }
                    .glassCard(padding: 0, cornerRadius: 18)

                    // Output
                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            Label("Output", systemImage: "doc.text")
                                .font(.headline)
                            HStack {
                                TextField("timeline", text: $outputFileName)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 200)
                                Text(".html")
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                    }
                    .glassCard(padding: 0, cornerRadius: 18)
                }
                .padding(.horizontal, 4)
            }
            .padding(.vertical, 24)

            // Bottom actions
            HStack(spacing: 16) {
                GlassButton(title: "Back", icon: "chevron.left") {
                    appState.screen = .importVideo
                }

                GlassButton(title: "Start Analysis", icon: "play.fill", isPrimary: true) {
                    startAnalysis()
                }
            }
            .padding(.bottom, 36)
        }
        .padding(.horizontal, 48)
    }

    private func startAnalysis() {
        let outputPath = NSTemporaryDirectory() + "\(outputFileName).html"
        appState.outputPath = outputPath
        appState.screen = .analyzing
    }
}
