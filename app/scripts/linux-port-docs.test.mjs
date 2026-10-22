// Run: node scripts/linux-port-docs.test.mjs
// Public port documents must name real repository files and describe the backend the tree actually holds.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { lstatSync, readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const app = path.resolve(here, '..')
const root = path.dirname(app)
const docs = path.join(app, 'docs', 'linux-port')
const names = readdirSync(docs).filter((name) => name.endsWith('.md'))
assert.ok(names.length > 0, 'the Linux port documents are there')

/** The accounts a text names on GitHub, from a full URL and from the short "owner/repository" form. */
const accountsNamed = (text) => {
  const out = new Set()
  for (const match of text.matchAll(/github\.com\/([A-Za-z0-9-]+)\//g)) out.add(match[1].toLowerCase())
  for (const match of text.matchAll(/(?<![\w.-])([A-Za-z0-9-]+)\/tensorfold\b/gi)) out.add(match[1].toLowerCase())
  return out
}

// The install/release URLs identify the project owner; shell paths and other GitHub links do not.
const readme = readFileSync(path.join(root, 'README.md'), 'utf8')
const releaseOwners = readme.matchAll(/https:\/\/github\.com\/([A-Za-z0-9-]+)\/TensorFold\/releases(?:\/|[?#\s)"`]|$)/gi)
const repositoryAccounts = new Set([...releaseOwners].map((match) => match[1].toLowerCase()))
assert.equal(repositoryAccounts.size, 1, 'the README must name one canonical release repository')

/** File references in simple shell instructions, with cd applied separately within each fenced block. */
const shellTokens = (text) => {
  const out = []
  for (const block of text.matchAll(/```(?:sh|bash|shell)\n([\s\S]*?)```/g)) {
    let cwd = root
    for (const line of block[1].split('\n')) {
      const code = line.split(/(?<!["'\w])#/)[0]
      for (const command of code.split(/&&|;/)) {
        const words = command.trim().split(/\s+/).map((word) => word.replace(/^["'`]+|["'`),;]+$/g, ''))
        if (words[0] === 'cd' && words[1]) {
          out.push({ token: words[1], cwd, directory: true })
          cwd = path.resolve(cwd, words[1])
        } else for (const token of words) out.push({ token, cwd, directory: false })
      }
    }
  }
  return out
}

/** The tokens of a section's prose, with the code blocks left out: they are checked as instructions. */
const sectionTokens = (text, heading) => {
  const start = text.indexOf(heading)
  assert.notEqual(start, -1, `the section "${heading}" is there`)
  const rest = text.slice(start + heading.length)
  const end = rest.search(/^## /m)
  const section = (end === -1 ? rest : rest.slice(0, end)).replace(/```[\s\S]*?```/g, ' ')
  const out = []
  for (const span of section.matchAll(/`([^`]+)`/g)) out.push(...span[1].split(/\s+/))
  return out
}

/** Whether a token can name a file in this repository. */
const namesAFile = (token) => {
  if (!token || /[<>@]/.test(token) || token.includes('://')) return false
  if (token.startsWith('~') || token.startsWith('/')) return false
  if (/^\d/.test(token)) return false
  if (!token.includes('/') && !/\.[A-Za-z0-9]+$/.test(token)) return false
  return true
}

/** Whether the repository holds the file a token names. */
const holds = (token, cwd = app, directory = false) => {
  const relative = path.relative(root, path.resolve(cwd, token))
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) return false
  try {
    let current = root
    let info = lstatSync(current)
    for (const part of relative.split(path.sep).filter(Boolean)) {
      current = path.join(current, part)
      info = lstatSync(current)
      if (info.isSymbolicLink()) return false
    }
    return directory ? info.isDirectory() : info.isFile() || info.isDirectory()
  } catch {
    return false
  }
}

test('document paths must exist at their exact location and working directory', () => {
  assert.equal(holds('missing/scripts/linux-port-docs.test.mjs', app), false)
  assert.equal(holds('scripts/linux-port-docs.test.mjs', app), true)
  assert.equal(holds('app/scripts/linux-port-docs.test.mjs', path.dirname(app)), true)
  assert.equal(holds('scripts/linux-port-docs.test.mjs', path.dirname(app)), false)
  assert.equal(holds('../README.md', app), true)
  assert.equal(holds('../../README.md', app), false)
  assert.equal(holds('package.json', app, true), false)
  const commands = '```sh\ncd app && node scripts/linux-port-docs.test.mjs\ncd mobile && cat package.json\n```'
  const references = shellTokens(commands).filter((item) => item.directory || namesAFile(item.token))
  assert.equal(references.length, 4)
  assert.equal(references.every((item) => holds(item.token, item.cwd, item.directory)), true)
  assert.equal(references.at(-1).cwd, path.join(app, 'mobile'))
})

test('the project account comes from the release repository, not shell path tokens', () => {
  assert.equal(repositoryAccounts.size, 1)
  for (const token of ['bin', 'pwd', 'v']) assert.equal(repositoryAccounts.has(token), false)
  const readme = readFileSync(path.join(app, '..', 'README.md'), 'utf8')
  const owner = /https:\/\/github\.com\/([A-Za-z0-9-]+)\/TensorFold\/releases\//.exec(readme)?.[1]
  assert.ok(owner)
  assert.equal(repositoryAccounts.has(owner.toLowerCase()), true)
})

for (const name of names) {
  const text = readFileSync(path.join(docs, name), 'utf8')

  // 1: a public document carries no home directory of one person.
  assert.deepEqual(
    text.match(/\/home\/[A-Za-z0-9._-]+/g) ?? [],
    [],
    `${name} must not carry a home directory path`,
  )

  // 2: a public document names no account but the repository's own.
  for (const account of accountsNamed(text)) {
    assert.ok(
      repositoryAccounts.has(account),
      `${name} must not name a personal account (${account}); name the repository's own instead`,
    )
  }

  // 3: the instructions name only files this repository holds.
  const instructions = shellTokens(text).filter((item) => item.directory || namesAFile(item.token))
  assert.deepEqual(
    instructions.filter((item) => !holds(item.token, item.cwd, item.directory)),
    [],
    `${name} must not tell the reader to use a file this repository does not hold`,
  )
}

// 4: the baseline document's setup section names only files this repository holds.
const baseline = readFileSync(path.join(docs, 'PHASE1-BASELINE.md'), 'utf8')
const setup = sectionTokens(baseline, '## Environment setup').filter(namesAFile)
assert.deepEqual(
  setup.filter((token) => !holds(token)),
  [],
  'the baseline document must not name a setup file this repository does not hold',
)

// 5: what the documents claim about the bubblewrap backend matches the code.
const backend = path.join(app, 'electron', 'agent', 'linux-sandbox.ts')
const claimsBackend = names.some((name) => {
  const text = readFileSync(path.join(docs, name), 'utf8')
  return /bubblewrap backend is (implemented|in this change|part of this change)/.test(text)
})
assert.equal(
  claimsBackend,
  existsInTree(backend),
  'the documents may claim the bubblewrap backend only while the repository holds it',
)

function existsInTree(file) {
  try {
    readFileSync(file)
    return true
  } catch {
    return false
  }
}

console.log(`Linux port documents: ${names.length} files, no personal paths, no files we do not hold`)
