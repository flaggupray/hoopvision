#!/bin/bash
set -e

echo "🏀 Building HoopVision macOS App..."
cd "$(dirname "$0")"

# Build with Swift Package Manager
swift build -c release 2>&1

# Create .app bundle
APP_DIR="./HoopVision.app"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

# Info.plist
cat > "$APP_DIR/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>HoopVision</string>
    <key>CFBundleIdentifier</key>
    <string>com.hoopvision.app</string>
    <key>CFBundleName</key>
    <string>HoopVision</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

# Find and copy binary
BIN_PATH=$(swift build --show-bin-path -c release 2>/dev/null || swift build --show-bin-path)
cp "$BIN_PATH/HoopVision" "$APP_DIR/Contents/MacOS/HoopVision"

echo "✅ App built: $APP_DIR"
echo "   Run: open $APP_DIR"
