import SwiftUI

@main
struct HoopVisionApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 960, minHeight: 680)
                .preferredColorScheme(.light)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1100, height: 780)
        .windowResizability(.contentSize)

        Settings {
            SettingsView()
                .environmentObject(appState)
        }
    }
}

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
