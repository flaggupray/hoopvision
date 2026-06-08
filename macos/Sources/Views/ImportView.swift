import SwiftUI
import UniformTypeIdentifiers

struct ImportView: View {
    @EnvironmentObject var appState: AppState
    @State private var isTargeted = false
    @State private var showFilePicker = false

    var body: some View {
        VStack(spacing: 28) {
            // Header
            VStack(spacing: 8) {
                Text("Import Video")
                    .font(.system(size: 28, weight: .bold, design: .rounded))

                Text("Drag and drop a basketball game video or select from your files")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 40)

            // Drop zone
            ZStack {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay {
                        RoundedRectangle(cornerRadius: 28, style: .continuous)
                            .stroke(
                                isTargeted ? Color.orange : Color.black.opacity(0.10),
                                lineWidth: isTargeted ? 2.5 : 1.0
                            )
                    }
                    .shadow(color: .black.opacity(0.04), radius: 12, y: 6)

                VStack(spacing: 20) {
                    Image(systemName: isTargeted ? "film.stack.fill" : "arrow.down.doc.fill")
                        .font(.system(size: 52))
                        .foregroundStyle(isTargeted ? .orange : .secondary)
                        .scaleEffect(isTargeted ? 1.15 : 1.0)
                        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isTargeted)

                    VStack(spacing: 6) {
                        Text(isTargeted ? "Drop to Import" : "Drop Video File Here")
                            .font(.title3)
                            .fontWeight(.semibold)

                        Text("Supports .mp4, .mov, .avi")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    if appState.selectedVideoPath.isEmpty {
                        Text("or")
                            .font(.caption)
                            .foregroundStyle(.tertiary)

                        Button("Choose File...") {
                            showFilePicker = true
                        }
                        .buttonStyle(.borderless)
                    } else {
                        HStack(spacing: 8) {
                            Image(systemName: "film")
                                .foregroundStyle(.orange)
                            Text((appState.selectedVideoPath as NSString).lastPathComponent)
                                .font(.callout)
                                .foregroundStyle(.primary)

                            Button(action: { appState.selectedVideoPath = "" }) {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                }
            }
            .frame(maxWidth: 560, minHeight: 280)
            .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
                handleDrop(providers: providers)
                return true
            }
            .fileImporter(
                isPresented: $showFilePicker,
                allowedContentTypes: [.mpeg4Movie, .quickTimeMovie, .avi],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let files) = result, let file = files.first {
                    appState.selectedVideoPath = file.path(percentEncoded: false)
                }
            }

            // Bottom actions
            HStack(spacing: 16) {
                GlassButton(title: "Back", icon: "chevron.left") {
                    appState.reset()
                }

                GlassButton(title: "Continue", icon: "arrow.right", isPrimary: true) {
                    appState.screen = .configure
                }
                .disabled(appState.selectedVideoPath.isEmpty)
            }
            .padding(.bottom, 40)
        }
        .padding(48)
    }

    private func handleDrop(providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, error in
            if let data = item as? Data,
               let url = URL(dataRepresentation: data, relativeTo: nil) {
                DispatchQueue.main.async {
                    appState.selectedVideoPath = url.path(percentEncoded: false)
                }
            }
        }
    }
}
