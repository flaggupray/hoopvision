import SwiftUI

struct WelcomeView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            // Logo and title
            VStack(spacing: 16) {
                ZStack {
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(LinearGradient(
                            colors: [.orange, Color.orange.opacity(0.75)],
                            startPoint: .topLeading, endPoint: .bottomTrailing
                        ))
                        .frame(width: 76, height: 76)
                        .shadow(color: .orange.opacity(0.35), radius: 16, y: 8)

                    Image(systemName: "basketball.fill")
                        .font(.system(size: 34))
                        .foregroundColor(.white)
                }

                VStack(spacing: 6) {
                    Text("HoopVision")
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)

                    Text("Basketball Game AI Analyzer")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
            }

            // Feature highlights
            HStack(spacing: 40) {
                FeatureItem(icon: "camera.fill", title: "Import Video", desc: "Drag & drop or select a basketball game recording")
                FeatureItem(icon: "brain.head.profile", title: "AI Analysis", desc: "YOLO detection + jersey number OCR + action classification")
                FeatureItem(icon: "list.clipboard", title: "Timeline", desc: "Narrative game timeline with human touch")
            }

            Spacer()

            // CTA button
            Button(action: { appState.screen = .importVideo }) {
                Text("Analyze a Game")
                    .font(.system(size: 17, weight: .semibold))
                    .padding(.horizontal, 36)
                    .padding(.vertical, 14)
            }
            .buttonStyle(GlassButtonStyle(isPrimary: true))

            Spacer()
        }
        .padding(60)
    }
}

struct FeatureItem: View {
    let icon: String
    let title: String
    let desc: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 28))
                .foregroundStyle(.orange)
                .frame(height: 36)

            Text(title)
                .font(.system(size: 14, weight: .semibold))

            Text(desc)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(width: 160)
        }
    }
}
