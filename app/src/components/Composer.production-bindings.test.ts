import assert from 'node:assert/strict'
import path from 'node:path'
import ts from 'typescript'

const root = process.cwd()
const targets = [
  'src/components/Composer.tsx',
  'src/components/composer/useComposerState.ts',
  'src/components/composer/useComposerWorkspace.ts',
  'src/components/composer/useComposerSubmit.tsx',
  'src/components/composer/ComposerView.tsx',
].map((file) => path.resolve(root, file))
const config = ts.readConfigFile(path.join(root, 'tsconfig.json'), ts.sys.readFile)
assert.ok(!config.error, config.error && ts.flattenDiagnosticMessageText(config.error.messageText, '\n'))
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root, { noEmit: true }, path.join(root, 'tsconfig.json'))
const program = ts.createProgram({ rootNames: targets, options: parsed.options })
const targetSet = new Set(targets)
const missing = ts.getPreEmitDiagnostics(program).filter((diagnostic) => {
  if (!diagnostic.file || !targetSet.has(diagnostic.file.fileName) || diagnostic.code !== 2304) return false
  return /^Cannot find name '(mode|thread)'[.]/.test(ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'))
})
const detail = missing.map((diagnostic) => {
  const at = diagnostic.file!.getLineAndCharacterOfPosition(diagnostic.start!)
  return `${path.basename(diagnostic.file!.fileName)}:${at.line + 1}:${at.character + 1}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`
}).join('\n')
assert.equal(missing.length, 0, `The real Composer modules must bind their original mode and thread props:\n${detail}`)
console.log('Production Composer modules typecheck without unbound mode or thread references')
