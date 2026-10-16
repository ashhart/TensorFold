// Run: ./node_modules/.bin/jiti electron/agent/slash-commands.test.ts
import assert from 'node:assert/strict'
import {
  loadCommands,
  parseCommandDocument,
  expandCommandArgs,
  tokenizeArgs,
  DEFAULT_COMMAND_MAX_BYTES,
  type CommandLoaderIO,
} from './slash-commands'

let passed = 0
function ok(label: string, cond: boolean) {
  assert.ok(cond, label)
  passed++
}

// ---- an in-memory IO fake so the loader never touches disk ----
function fakeIO(files: Record<string, string>, mtimes: Record<string, number> = {}): CommandLoaderIO {
  return {
    async readdir() {
      return Object.keys(files)
    },
    async readFile(filePath: string) {
      const name = filePath.split('/').pop()!
      if (!(name in files)) throw new Error('ENOENT')
      return files[name]
    },
    async stat(filePath: string) {
      const name = filePath.split('/').pop()!
      if (!(name in files)) throw new Error('ENOENT')
      return { size: Buffer.byteLength(files[name]), mtimeMs: mtimes[name] ?? 1000, isFile: true }
    },
  }
}

async function main() {
  // --- parseCommandDocument: front matter + body template ---
  {
    const doc = parseCommandDocument(
      ['---', 'name: Fix Bug', 'description: Investigate and fix.', 'args: [file, mode]', '---', '', 'Fix $1 in $2 mode. Full: $ARGUMENTS'].join('\n'),
      'fix-bug.md',
    )
    ok('id from filename', doc.id === 'fix-bug')
    ok('name from front matter', doc.name === 'Fix Bug')
    ok('description parsed', doc.description === 'Investigate and fix.')
    ok('inline args list parsed', doc.args.length === 2 && doc.args[0].name === 'file' && doc.args[1].name === 'mode')
    ok('template is the body', doc.template.startsWith('Fix $1 in $2 mode.'))
  }

  // --- fallback: no front matter -> humanised name, body is template ---
  {
    const doc = parseCommandDocument('Summarise the diff and $ARGUMENTS', 'summarise_diff.md')
    ok('humanised name fallback', doc.name === 'Summarise Diff')
    ok('no args declared', doc.args.length === 0)
    ok('template = whole body', doc.template === 'Summarise the diff and $ARGUMENTS')
  }

  // --- prompt: front-matter field overrides body ---
  {
    const doc = parseCommandDocument(
      ['---', 'name: P', 'prompt: Do the thing with $1', '---', 'body ignored'].join('\n'),
      'p.md',
    )
    ok('prompt field wins over body', doc.template === 'Do the thing with $1')
  }

  // --- block scalar prompt (multi-line) ---
  {
    const doc = parseCommandDocument(
      ['---', 'name: Multi', 'prompt: |', '  line one', '  line two $1', '---'].join('\n'),
      'multi.md',
    )
    ok('block-scalar prompt preserved', doc.template === 'line one\nline two $1')
  }

  // --- nested args list with required + description ---
  {
    const doc = parseCommandDocument(
      ['---', 'name: N', 'args:', '  - name: target', '    required: true', '  - name: extra', '---', 'x'].join('\n'),
      'n.md',
    )
    ok('nested args parsed', doc.args.length === 2)
    ok('required flag honoured', doc.args[0].name === 'target' && doc.args[0].required === true)
    ok('second arg optional', doc.args[1].name === 'extra' && doc.args[1].required === false)
  }

  // --- expandCommandArgs: positional, $ARGUMENTS, named, $$ literal, unknown left alone ---
  {
    const spec = { template: 'Fix $1 ($file) in $2 mode; all=[$ARGUMENTS]; cost $$5; keep $unknown', args: [{ name: 'file', required: true }, { name: 'mode', required: false }] }
    const { prompt, missing } = expandCommandArgs(spec, 'src/app.ts fast')
    ok('positional $1', prompt.includes('Fix src/app.ts'))
    ok('named $file resolves to first positional', prompt.includes('(src/app.ts)'))
    ok('positional $2', prompt.includes('in fast mode'))
    ok('$ARGUMENTS is the raw string', prompt.includes('all=[src/app.ts fast]'))
    ok('$$ becomes literal dollar', prompt.includes('cost $5'))
    ok('unknown placeholder left verbatim', prompt.includes('keep $unknown'))
    ok('nothing missing', missing.length === 0)
  }

  // --- missing required arg reported ---
  {
    const spec = { template: '$1 $2', args: [{ name: 'a', required: true }, { name: 'b', required: true }] }
    const { missing } = expandCommandArgs(spec, 'only-one')
    ok('missing required reported', missing.length === 1 && missing[0] === 'b')
  }

  // --- brace form ${1} and out-of-range positional -> empty ---
  {
    const spec = { template: '<${1}x><$3>', args: [] as { name: string; required: boolean }[] }
    const { prompt } = expandCommandArgs(spec, 'hello')
    ok('brace form ${1}', prompt.startsWith('<hellox>'))
    ok('out-of-range positional -> empty', prompt.includes('<>'))
  }

  // --- tokenizer honours quotes ---
  {
    assert.deepEqual(tokenizeArgs('a "b c" \'d e\' f'), ['a', 'b c', 'd e', 'f'])
    ok('quoted tokens preserved', true)
    assert.deepEqual(tokenizeArgs('  spaced   out  '), ['spaced', 'out'])
    ok('collapses whitespace', true)
    assert.deepEqual(tokenizeArgs('"a\\"b"'), ['a"b'])
    ok('escaped quote inside double', true)
    assert.deepEqual(tokenizeArgs('esc"a\\"b"'), ['esca"b'])
    ok('adjacent unquoted+quoted segments concatenate', true)
  }

  // --- loadCommands: full directory scan via fake IO ---
  {
    const io = fakeIO(
      {
        'alpha.md': '---\nname: Alpha\n---\nrun alpha $1',
        'beta.md': '# Beta\n\nrun beta',
        'notes.txt': 'ignored non-md',
        'empty.md': '---\nname: Empty\n---\n', // no body, no description -> skipped
      },
      { 'alpha.md': 5000, 'beta.md': 6000 },
    )
    const { commands, errors } = await loadCommands('/repo/.vontra/commands', io)
    ok('only .md files loaded (txt ignored)', commands.every((c) => c.id !== 'notes'))
    ok('empty command skipped with error', errors.some((e) => e.file === 'empty.md'))
    ok('two valid commands', commands.length === 2)
    ok('sorted by id', commands[0].id === 'alpha' && commands[1].id === 'beta')
    ok('mtime captured', commands[0].updatedAt === 5000)
    ok('sourcePath set', commands[0].sourcePath === '/repo/.vontra/commands/alpha.md')
  }

  // --- loadCommands: missing dir -> empty, no throw ---
  {
    const throwingIO: CommandLoaderIO = {
      async readdir() { throw new Error('ENOENT') },
      async readFile() { throw new Error('nope') },
      async stat() { throw new Error('nope') },
    }
    const res = await loadCommands('/nope', throwingIO)
    ok('missing dir -> empty commands', res.commands.length === 0 && res.errors.length === 0)
  }

  // --- loadCommands: duplicate id -> first wins, rest error ---
  {
    const io = fakeIO({ 'dup.md': 'first', 'DUP.md': 'second' })
    const { commands, errors } = await loadCommands('/repo', io)
    ok('duplicate id collapses to one', commands.length === 1)
    ok('duplicate reported as error', errors.some((e) => /Duplicate/.test(e.error)))
  }

  // --- loadCommands: oversized file skipped ---
  {
    const big = 'x'.repeat(DEFAULT_COMMAND_MAX_BYTES + 1)
    const io = fakeIO({ 'big.md': big })
    const { commands, errors } = await loadCommands('/repo', io)
    ok('oversized file skipped', commands.length === 0 && errors.some((e) => /exceeds/.test(e.error)))
  }

  console.log(`slash-commands.test.ts: ${passed} assertions passed`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
