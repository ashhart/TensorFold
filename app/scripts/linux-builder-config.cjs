const { build: shared } = require('../package.json')

// Loaded as an explicit config file, not an overlay on the Mac release configuration.
module.exports = {
  extends: null,
  appId: shared.appId,
  productName: shared.productName,
  npmRebuild: shared.npmRebuild,
  directories: { ...shared.directories, output: 'release/linux' },
  files: [...shared.files, '!node_modules/**/*.pdb',
    '!node_modules/onnxruntime-node/bin/napi-v3/{darwin,win32}/**/*',
    `!node_modules/onnxruntime-node/bin/napi-v3/linux/${process.arch === 'x64' ? 'arm64' : 'x64'}/**/*`],
  extraResources: shared.extraResources.filter(({ to }) => ['engine', 'voice-notices', 'third-party-notices'].includes(to)),
  asar: shared.asar,
  afterPack: shared.afterPack,
  // Linux development artifacts are unsigned; the Mac signing requirement remains untouched.
  forceCodeSigning: false,
  electronFuses: { ...shared.electronFuses },
  asarUnpack: ['node_modules/onnxruntime-node/bin/napi-v3/linux/${arch}/**/*'],
  publish: null,
  toolsets: { appimage: '1.0.3' },
  appImage: { executableArgs: [] },
  deb: {
    maintainer: 'TensorFold', priority: 'optional',
    // Electron links ALSA even when the application starts without a desktop window.
    depends: ['bubblewrap', 'libgtk-3-0', 'libnotify4', 'libnss3', 'libxss1', 'libxtst6', 'xdg-utils',
      'libatspi2.0-0', 'libuuid1', 'libsecret-1-0', 'libasound2t64 | libasound2'],
  },
  linux: {
    syncDesktopName: true,
    target: [{ target: 'AppImage', arch: ['x64'] }],
    artifactName: '${productName}-${version}-linux-${arch}.${ext}',
    category: 'Development',
    icon: 'build/icon.png',
  },
}
