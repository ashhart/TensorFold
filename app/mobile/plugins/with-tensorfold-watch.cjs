const fs = require('node:fs')
const path = require('node:path')
const { withXcodeProject } = require('@expo/config-plugins')
const NAME = 'TensorFoldWatch'
const FILES = ['TensorFoldWatchApp.swift', 'WatchWorkspace.swift', 'WatchProtocol.swift', 'WatchAudioPlayer.swift']
function ensureWatchTargetDependency(project, companionId, watchId) {
  const objects = project.hash.project.objects
  const companion = objects.PBXNativeTarget[companionId]
  if (!companion || !objects.PBXNativeTarget[watchId]) throw new Error('TensorFold watch requires an existing companion target.')
  if (companionId === watchId) throw new Error('TensorFold companion and watch targets must be distinct.')
  // xcode.addTargetDependency silently does nothing if either section is absent,
  // as happens in a fresh Expo project without extensions or test targets.
  objects.PBXTargetDependency ||= {}
  objects.PBXContainerItemProxy ||= {}
  companion.dependencies ||= []
  if (!companion.dependencies.some(entry => objects.PBXTargetDependency[entry.value]?.target === watchId)) {
    project.addTargetDependency(companionId, [watchId])
  }
}
function configureWatchProject(project, platformRoot, config, sourceRoot) {
  const companion = config.ios?.bundleIdentifier
  if (!companion || !/^[A-Za-z0-9.-]+$/.test(companion)) throw new Error('TensorFold watch requires a valid companion iOS bundle identifier.')
  const existing = Object.entries(project.pbxNativeTargetSection()).find(([id, target]) => !id.endsWith('_comment') && String(target.name).replaceAll('"', '') === NAME)
  let target
  if (existing) target = {uuid: existing[0], pbxNativeTarget: existing[1]}
  else target = project.addTarget(NAME, 'watch2_app', NAME, `${companion}.watchkitapp`)
  // The watch2 helper supplies embedding; ensure its build dependency also exists.
  ensureWatchTargetDependency(project, project.getFirstTarget().uuid, target.uuid)
  target.pbxNativeTarget.productType = '"com.apple.product-type.application"'
  const folder = path.join(platformRoot, NAME)
  fs.mkdirSync(folder, {recursive: true})
  for (const file of FILES) fs.copyFileSync(path.resolve(sourceRoot, '..', '..', 'watchOS', file), path.join(folder, file))
  const escape = (value) => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  fs.writeFileSync(path.join(folder, 'Info.plist'), `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleDisplayName</key><string>TensorFold</string>
<key>CFBundleExecutable</key><string>$(EXECUTABLE_NAME)</string>
<key>CFBundleIdentifier</key><string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
<key>CFBundleName</key><string>$(PRODUCT_NAME)</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>${escape(config.version || '1.0.0')}</string>
<key>CFBundleVersion</key><string>${escape(config.ios?.buildNumber || '1')}</string>
<key>WKApplication</key><true/>
<key>WKCompanionAppBundleIdentifier</key><string>${escape(companion)}</string>
<key>WKRunsIndependentlyOfCompanionApp</key><false/>
</dict></plist>\n`)
  const sections = project.hash.project.objects
  const list = sections.XCConfigurationList[target.pbxNativeTarget.buildConfigurationList]
  for (const entry of list.buildConfigurations) {
    const settings = sections.XCBuildConfiguration[entry.value].buildSettings
    Object.assign(settings, {PRODUCT_BUNDLE_IDENTIFIER: `"${companion}.watchkitapp"`, PRODUCT_NAME: `"${NAME}"`, SDKROOT: 'watchos', WATCHOS_DEPLOYMENT_TARGET: '9.0',
      SUPPORTED_PLATFORMS: '"watchos watchsimulator"', TARGETED_DEVICE_FAMILY: '4', SWIFT_VERSION: '5.0', INFOPLIST_FILE: `"${NAME}/Info.plist"`, GENERATE_INFOPLIST_FILE: 'NO',
      LD_RUNPATH_SEARCH_PATHS: '"$(inherited) @executable_path/Frameworks"', CODE_SIGN_STYLE: 'Automatic', SKIP_INSTALL: 'YES',
      CURRENT_PROJECT_VERSION: `"${config.ios?.buildNumber || '1'}"`, MARKETING_VERSION: `"${config.version || '1.0.0'}"`, ENABLE_USER_SCRIPT_SANDBOXING: 'YES'})
    if (config.ios?.appleTeamId) settings.DEVELOPMENT_TEAM = `"${config.ios.appleTeamId}"`
  }
  if (!existing) {
    project.addBuildPhase(FILES.map((file) => `${NAME}/${file}`), 'PBXSourcesBuildPhase', 'Sources', target.uuid)
    project.addBuildPhase([], 'PBXFrameworksBuildPhase', 'Frameworks', target.uuid)
    project.addBuildPhase([], 'PBXResourcesBuildPhase', 'Resources', target.uuid)
    const group = project.addPbxGroup([...FILES.map((file) => `${NAME}/${file}`), `${NAME}/Info.plist`], NAME, '""')
    project.addToPbxGroup(group.uuid, project.getFirstProject().firstProject.mainGroup)
  }
  return target.uuid
}
module.exports = function withTensorFoldWatch(config) {
  return withXcodeProject(config, (mod) => {
    configureWatchProject(mod.modResults, mod.modRequest.platformProjectRoot, mod, mod.modRequest.projectRoot)
    return mod
  })
}
module.exports.configureWatchProject = configureWatchProject
module.exports.ensureWatchTargetDependency = ensureWatchTargetDependency
