import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { insideWorkspace, scanWorkspace, workspaceFileBytes } from './isolated-workspace-files'
import { localWindowsPath, windowsCommandWords } from './windows-sandbox-paths'
import { windowsLocalInput } from './windows-local-input'
import { snapshotWindowsWorkspace } from './windows-workspace-snapshot'
import type { IsolatedWorkspace } from './isolated-workspace'

export interface WindowsRuntimeView {
  command: string
  args: string[]
  env: NodeJS.ProcessEnv
  readViews: Array<{ source: string; target: string }>
}

export async function windowsRuntimeView(
  layout: IsolatedWorkspace, command: string, args: string[], inputEnv: NodeJS.ProcessEnv, allowPathRuntimes = false,
): Promise<WindowsRuntimeView> {
  const windows = await fs.realpath(process.env.SystemRoot || 'C:\\Windows')
  const home = await fs.realpath(os.homedir())
  const mappings = new Map<string, string>()
  let bytes = 0, count = 0
  const copyRoot = async (input: string): Promise<void> => {
    let source: string
    const local = localWindowsPath(input)
    if (!await windowsLocalInput(local)) return
    try { source = await fs.realpath(local) } catch { return }
    if (insideWorkspace(windows, source) || insideWorkspace(layout.root, source)) return
    if (insideWorkspace(source, home) || insideWorkspace(source, layout.container) || source === path.parse(source).root) {
      throw new Error('A Windows runtime directory would expose a private home or isolation ancestor.')
    }
    if ([...mappings.keys()].some(root => insideWorkspace(root, source))) return
    if (mappings.size >= 32) throw new Error('Too many Windows runtime roots.')
    const target = path.join(layout.container, 'runtimes', String(mappings.size))
    await fs.mkdir(target, { recursive: true, mode: 0o700 })
    await snapshotWindowsWorkspace(source, target, true)
    const inventory = await scanWorkspace(target)
    count += inventory.size
    for (const entry of inventory.values()) if (entry.kind === 'file') bytes += entry.size
    if (bytes > 2 * 1024 * 1024 * 1024 || count > 100000) throw new Error('Windows runtime copies exceed the bounded isolation limit.')
    mappings.set(source, target)
  }
  const directories: string[] = []
  for (const directory of (inputEnv.PATH ?? '').split(';').map(value => value.replace(/^"|"$/g, '')).filter(Boolean).map(localWindowsPath)) {
    if (await windowsLocalInput(directory)) directories.push(directory)
  }
  const runtimeRoot = async (directory: string): Promise<string> => {
    const parent = path.dirname(directory)
    if (await windowsLocalInput(path.join(parent, 'mingw64/bin/git.exe'))) return parent
    if (path.basename(directory).toLowerCase() === 'scripts'
      && await windowsLocalInput(path.join(parent, 'python.exe'))) return parent
    return directory
  }
  if (command.includes(':') && !/^[a-z]:[\\/]/i.test(command)) throw new Error('Windows sandbox commands cannot use drive-relative or stream paths.')
  const executable = path.isAbsolute(command) ? command : /[\\/]/.test(command) ? path.resolve(layout.root, command) : await resolveExecutable(command, directories)
  if (!executable) throw new Error('The Windows sandbox command could not be resolved.')
  localWindowsPath(executable)
  if (!insideWorkspace(windows, executable) && !insideWorkspace(layout.root, executable)) await copyRoot(await runtimeRoot(path.dirname(executable)))
  const shell = path.basename(executable).toLowerCase() === 'cmd.exe' && args.slice(0, 3).join(' ').toLowerCase() === '/d /s /c'
  if (shell) for (const word of windowsCommandWords(args[3] ?? '')) {
    const binary = await resolveExecutable(word, directories)
    if (binary) await copyRoot(await runtimeRoot(path.dirname(binary)))
  }
  if (allowPathRuntimes) for (const directory of directories) {
    if (/\\(?:WindowsApps|WinGet\\Links)$/i.test(directory)) continue
    if (await fs.stat(directory).then(stat => stat.isDirectory(), () => false)) await copyRoot(await runtimeRoot(directory))
  }
  for (const argument of args) {
    if (!/^[a-z]:[\\/]/i.test(argument)) continue
    localWindowsPath(argument)
    if (!insideWorkspace(layout.root, argument) && !insideWorkspace(windows, argument)
      && await windowsLocalInput(argument) && await fs.lstat(argument).then(stat => stat.isFile(), () => false)) {
      if (/\.(?:[cm]?js|py|rb|ps1|cmd|bat|exe)$/i.test(argument)) await copyRoot(path.dirname(argument))
      else {
        const target = path.join(layout.container, 'inputs', String(mappings.size), path.basename(argument))
        await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 })
        await fs.writeFile(target, await workspaceFileBytes(argument), { flag: 'wx' })
        mappings.set(argument, target)
      }
    }
  }
  const project = (input: string): string => {
    if (insideWorkspace(layout.root, input)) return path.join(layout.workspace, path.relative(layout.root, input))
    const matching = [...mappings].sort(([a], [b]) => b.length - a.length).find(([root]) => insideWorkspace(root, input))
    return matching ? path.join(matching[1], path.relative(matching[0], input)) : input
  }
  return {
    readViews: [...mappings].map(([source, target]) => ({ source, target })),
    command: project(executable), args: args.map(argument => path.isAbsolute(argument) ? project(argument) : argument),
    env: {
      ...inputEnv, PATH: directories.filter(directory => insideWorkspace(windows, directory) || insideWorkspace(layout.root, directory)
        || [...mappings.keys()].some(root => insideWorkspace(root, directory))).map(directory => project(directory)).join(';'),
      SystemRoot: windows, ComSpec: path.join(windows, 'System32/cmd.exe'), PATHEXT: '.COM;.EXE;.BAT;.CMD',
      HOME: layout.workspace, USERPROFILE: layout.workspace, APPDATA: layout.scratch, LOCALAPPDATA: layout.scratch,
      TMP: layout.scratch, TEMP: layout.scratch, TMPDIR: layout.scratch,
    },
  }
}

async function resolveExecutable(command: string, directories: string[]): Promise<string | null> {
  for (const directory of directories) for (const extension of path.extname(command) ? [''] : ['.exe', '.com', '.cmd', '.bat']) {
    const file = path.join(directory, command + extension)
    localWindowsPath(file)
    if (await fs.lstat(file).then(stat => stat.isFile() && !stat.isSymbolicLink(), () => false)) return file
  }
  return null
}
