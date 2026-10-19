const { build: shared } = require('../package.json')

// Loaded as an explicit config file, not an overlay on the Mac release configuration.
module.exports = {
  extends: null,
  appId: shared.appId,
  productName: shared.productName,
  npmRebuild: shared.npmRebuild,
  directories: { ...shared.directories, output: 'release/linux' },
  files: [...shared.files],
  extraResources: shared.extraResources.filter(({ to }) => ['voice-notices', 'third-party-notices'].includes(to)),
  asar: shared.asar,
  afterPack: shared.afterPack,
  // Linux development artifacts are unsigned; the Mac signing requirement remains untouched.
  forceCodeSigning: false,
  electronFuses: { ...shared.electronFuses },
  asarUnpack: ['node_modules/onnxruntime-node/bin/napi-v3/linux/x64/**/*'],
  publish: null,
  linux: {
    target: [{ target: 'AppImage', arch: ['x64'] }],
    artifactName: '${productName}-${version}-linux-${arch}.${ext}',
    category: 'Development',
    icon: 'build/icon.png',
  },
}
