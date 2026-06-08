import SwiftUI
import WebKit

struct TimelineResultView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(spacing: 0) {
            // Header toolbar
            HStack {
                GlassButton(title: "New Analysis", icon: "arrow.left") {
                    appState.startNewAnalysis()
                }

                Spacer()

                Text("Game Timeline")
                    .font(.headline)

                Spacer()

                HStack(spacing: 12) {
                    GlassButton(title: "Share", icon: "square.and.arrow.up") {
                        shareOutput()
                    }

                    GlassButton(title: "Open in Browser", icon: "safari") {
                        if !appState.outputPath.isEmpty {
                            NSWorkspace.shared.open(URL(fileURLWithPath: appState.outputPath))
                        }
                    }

                    GlassButton(title: "Export", icon: "arrow.down.doc", isPrimary: true) {
                        exportFile()
                    }
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial)
            .overlay(alignment: .bottom) {
                Divider()
            }

            // WebView showing the timeline HTML
            if appState.outputPath.isEmpty || !FileManager.default.fileExists(atPath: appState.outputPath) {
                VStack(spacing: 16) {
                    Image(systemName: "doc.questionmark")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                    Text("Timeline output not found")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                TimelineWebView(htmlPath: appState.outputPath)
                    .ignoresSafeArea(edges: .bottom)
            }
        }
    }

    private func shareOutput() {
        // Placeholder for share sheet
        let picker = NSOpenPanel()
        picker.message = "Timeline saved to: \(appState.outputPath)"
    }

    private func exportFile() {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.html]
        savePanel.nameFieldStringValue = "game-timeline.html"
        savePanel.begin { response in
            if response == .OK, let url = savePanel.url {
                try? FileManager.default.copyItem(
                    atPath: appState.outputPath,
                    toPath: url.path(percentEncoded: false)
                )
            }
        }
    }
}

// MARK: - WebView Wrapper

struct TimelineWebView: NSViewRepresentable {
    let htmlPath: String

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        let prefs = WKWebpagePreferences()
        prefs.allowsContentJavaScript = true
        config.defaultWebpagePreferences = prefs

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")

        let url = URL(fileURLWithPath: htmlPath)
        webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())

        return webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {
        let url = URL(fileURLWithPath: htmlPath)
        nsView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    }
}
