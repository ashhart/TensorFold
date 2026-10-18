// Build-only verification. No account, signing, simulator boot, device install or generated mobile/ios edits.
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
import { spawnSync } from 'node:child_process'
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(root, 'package.json'))
const output = process.argv[2]
if (!output || !path.isAbsolute(output) || output === root || output.startsWith(`${root}/`)) throw new Error('Supply an absolute private output directory outside the app source.')
await mkdir(output, {recursive: true})
const projectFolder = path.join(output, 'TensorFoldWatchQA.xcodeproj')
await mkdir(projectFolder, {recursive: true})
const projectPath = path.join(projectFolder, 'project.pbxproj')
// A minimal host target is enough to exercise the same config plugin target/embedding code.
await writeFile(projectPath, `// !$*UTF8*$!
{ archiveVersion = 1; classes = {}; objectVersion = 56; objects = {
A00000000000000000000001 = {isa = PBXProject; attributes = {}; buildConfigurationList = A00000000000000000000004; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base,); mainGroup = A00000000000000000000002; productRefGroup = A00000000000000000000003; projectDirPath = ""; projectRoot = ""; targets = (A00000000000000000000007 /* TensorFoldPhone */,);};
A00000000000000000000002 = {isa = PBXGroup; children = (A00000000000000000000003 /* Products */,); sourceTree = "<group>";};
A00000000000000000000003 = {isa = PBXGroup; children = (); name = Products; sourceTree = "<group>";};
A00000000000000000000004 = {isa = XCConfigurationList; buildConfigurations = (A00000000000000000000005 /* Debug */, A00000000000000000000006 /* Release */,); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;};
A00000000000000000000005 = {isa = XCBuildConfiguration; buildSettings = {}; name = Debug;};
A00000000000000000000006 = {isa = XCBuildConfiguration; buildSettings = {}; name = Release;};
A00000000000000000000007 = {isa = PBXNativeTarget; buildConfigurationList = A00000000000000000000004; buildPhases = (); buildRules = (); dependencies = (); name = TensorFoldPhone; productName = TensorFoldPhone; productType = "com.apple.product-type.application";};
}; rootObject = A00000000000000000000001; }
`)
const project = require('xcode').project(projectPath); project.parseSync()
// xcode's parser expects dictionary-section comments to index objects; normalize our fixture.
if (!project.hash.project.objects.PBXProject) {
  const original = project.hash.project.objects, sections = {}
  for (const [key, object] of Object.entries(original)) {
    if (!object || typeof object !== 'object' || !object.isa) continue
    sections[object.isa] ??= {}; sections[object.isa][key] = object; sections[object.isa][`${key}_comment`] = object.name || object.isa
  }
  project.hash.project.objects = sections
}
for (const name of ['PBXBuildFile', 'PBXFileReference', 'PBXContainerItemProxy', 'PBXTargetDependency', 'PBXSourcesBuildPhase', 'PBXResourcesBuildPhase', 'PBXFrameworksBuildPhase']) project.hash.project.objects[name] ??= {}
const config = JSON.parse(await readFile(path.join(root, 'app.json'), 'utf8')).expo
const {configureWatchProject} = require('./plugins/with-tensorfold-watch.cjs')
configureWatchProject(project, output, config, root)
const first = project.writeSync()
configureWatchProject(project, output, config, root)
if (project.writeSync() !== first) throw new Error('Watch config plugin is not idempotent.')
await writeFile(projectPath, project.writeSync())
const result = spawnSync('xcodebuild', ['-project', projectFolder, '-target', 'TensorFoldWatch', '-configuration', 'Debug', '-sdk', 'watchsimulator', `SYMROOT=${path.join(output, 'build')}`, `OBJROOT=${path.join(output, 'objects')}`, 'CODE_SIGNING_ALLOWED=NO', 'ARCHS=arm64', 'build'], {encoding: 'utf8', maxBuffer: 20 * 1024 * 1024})
await writeFile(path.join(output, 'build.log'), [result.stdout, result.stderr].filter(Boolean).join('\n'))
console.log(result.status === 0 ? 'Unsigned watch simulator build passed; no simulator or device was launched.' : `Watch build failed; inspect ${path.join(output, 'build.log')}`)
if (result.status !== 0) process.exit(result.status || 1)
