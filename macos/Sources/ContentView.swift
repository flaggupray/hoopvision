import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        ZStack {
            // Ambient background
            GlassBackground()

            switch appState.screen {
            case .welcome:
                WelcomeView()
            case .importVideo:
                ImportView()
            case .configure:
                ConfigView()
            case .analyzing:
                AnalysisProgressView()
            case .result:
                TimelineResultView()
            }
        }
        .animation(.spring(response: 0.5, dampingFraction: 0.8), value: appState.screen)
    }
}

// MARK: - Glass Card modifier

struct GlassCard: ViewModifier {
    var padding: CGFloat = 24
    var cornerRadius: CGFloat = 28

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(.regularMaterial)
                    .shadow(color: .black.opacity(0.06), radius: 20, y: 6)
                    .shadow(color: .black.opacity(0.03), radius: 3, y: 1)
            }
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(.white.opacity(0.5), lineWidth: 0.5)
            }
    }
}

extension View {
    func glassCard(padding: CGFloat = 24, cornerRadius: CGFloat = 28) -> some View {
        modifier(GlassCard(padding: padding, cornerRadius: cornerRadius))
    }
}

// MARK: - Glass Background

struct GlassBackground: View {
    var body: some View {
        ZStack {
            Color(red: 0.95, green: 0.95, blue: 0.97)

            Circle()
                .fill(LinearGradient(
                    colors: [.orange.opacity(0.12), .clear],
                    startPoint: .topTrailing, endPoint: .bottomLeading
                ))
                .frame(width: 500, height: 500)
                .offset(x: 200, y: -200)
                .blur(radius: 80)

            Circle()
                .fill(LinearGradient(
                    colors: [.blue.opacity(0.10), .clear],
                    startPoint: .bottomLeading, endPoint: .topTrailing
                ))
                .frame(width: 400, height: 400)
                .offset(x: -180, y: 180)
                .blur(radius: 80)
        }
        .ignoresSafeArea()
    }
}

// MARK: - Navigation Button

struct GlassButton: View {
    let title: String
    let icon: String
    var isPrimary: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: icon)
                .font(.system(size: 14, weight: .semibold))
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
        }
        .buttonStyle(GlassButtonStyle(isPrimary: isPrimary))
    }
}

struct GlassButtonStyle: ButtonStyle {
    var isPrimary: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background {
                if isPrimary {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(.orange)
                        .shadow(color: .orange.opacity(0.3), radius: 8, y: 4)
                } else {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(.ultraThinMaterial)
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .stroke(.black.opacity(0.08), lineWidth: 0.5)
                        }
                }
            }
            .foregroundColor(isPrimary ? .white : .primary)
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}
