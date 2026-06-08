import SwiftUI
import AppKit

// MARK: - App Delegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        Task { @MainActor in
            NSApp.appearance = NSAppearance(named: .aqua)
        }
    }

    func applicationWillBecomeActive(_ notification: Notification) {
        Task { @MainActor in
            configureAllWindows()
        }
    }

    @MainActor
    private func configureAllWindows() {
        for window in NSApp.windows {
            window.titlebarAppearsTransparent = true
            window.isOpaque = false
            window.backgroundColor = NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1.0)
            window.hasShadow = true
            window.isMovableByWindowBackground = true
        }
    }
}

// MARK: - App

@main
struct HoopVisionApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 960, minHeight: 680)
                .background(Color(white: 0.955))
                .onAppear {
                    configureCurrentWindow()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1100, height: 780)
        .windowResizability(.contentSize)

        Settings {
            SettingsView()
                .environmentObject(appState)
        }
    }

    private func configureCurrentWindow() {
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 100_000_000)
            for window in NSApp.windows where window.title.contains("HoopVision")
                                        || window.isKeyWindow
                                        || window.isMainWindow {
                window.titlebarAppearsTransparent = true
                window.isOpaque = false
                window.backgroundColor = NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1.0)
                window.hasShadow = true
                window.isMovableByWindowBackground = true
            }
        }
    }
}

// MARK: - Settings

struct SettingsView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        TabView {
            Form {
                Section("AI Engine Path") {
                    TextField("Path to hoopvision/engine", text: $appState.enginePath)
                        .frame(width: 360)
                    Text("Path to the Python engine directory")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Section("Default Device") {
                    Picker("Inference Device", selection: $appState.defaultDevice) {
                        Text("Auto").tag("auto")
                        Text("CPU").tag("cpu")
                        Text("GPU (MPS)").tag("mps")
                    }
                }
            }
            .tabItem { Label("Engine", systemImage: "gearshape") }
            .padding()

            Form {
                Section("About") {
                    Text("HoopVision 0.1.0")
                        .font(.headline)
                    Text("Basketball game AI analyzer")
                        .foregroundStyle(.secondary)
                }
            }
            .tabItem { Label("About", systemImage: "info.circle") }
            .padding()
        }
        .frame(width: 460, height: 300)
    }
}
