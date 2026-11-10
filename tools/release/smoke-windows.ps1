#requires -Version 7.0
param([string]$Archive, [string]$Version, [switch]$AllowHostOnly)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-NativeProbe {
    param([string]$VersionText, [string]$HelpText, [string]$ServeText,
          [string]$CapabilitiesText, [string]$ExpectedVersion, [switch]$HostOnly)
    if ($VersionText.Trim() -cne "tensorfold-native $ExpectedVersion") { throw 'Native version protocol mismatch.' }
    if ($HelpText -notmatch '\bserve\s+MODEL\b' -or $ServeText -notmatch 'usage:\s+tensorfold\s+serve\b') {
        throw 'The packaged executable does not expose the native serving CLI.'
    }
    $caps = ConvertFrom-Json -AsHashtable $CapabilitiesText
    if ($caps['schema'] -ne 1 -or $caps['engine'] -cne 'zig' -or $caps['version'] -cne $ExpectedVersion) {
        throw 'Native capabilities identity mismatch.'
    }
    $flags = $caps['serve']['flags']
    if ($flags -isnot [System.Collections.IDictionary] -or -not $flags.Contains('context') -or -not $flags.Contains('parallel')) {
        throw 'Native serving capabilities are missing.'
    }
    if (-not $HostOnly) {
        if ($caps['backends'] -notcontains 'cuda' -or $caps['families'].Count -eq 0) {
            throw 'The Windows CUDA backend is not registered.'
        }
        $grammar = @($caps['optional_libraries'] | Where-Object { $_['soname'] -eq 'tfgrammar.dll' -and $_['found'] -eq $true })
        if ($grammar.Count -ne 1) { throw 'The packaged grammar runtime could not be loaded.' }
    }
}

function Expand-VerifiedWindowsArchive {
    param([string]$ArchivePath, [string]$ExpectedVersion, [switch]$HostOnlyAllowed)
    if ($ExpectedVersion -notmatch '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$') {
        throw 'A canonical release version is required.'
    }
    $archiveFile = [IO.Path]::GetFullPath($ArchivePath)
    $name = [IO.Path]::GetFileNameWithoutExtension($archiveFile)
    $normal = "tensorfold-$ExpectedVersion-windows-x86_64"
    if ($name -cne $normal -and $name -cne "$normal-host-only") { throw 'Unexpected Windows archive name.' }
    $hostOnly = $name.EndsWith('-host-only')
    if ($hostOnly -and -not $HostOnlyAllowed) { throw 'Host-only packages cannot qualify an inference release.' }
    $companion = [IO.File]::ReadAllText("$archiveFile.sha256").Trim()
    if ($companion -notmatch '^([0-9a-f]{64})\s+\*?([^/\\]+)$' -or $Matches[2] -cne [IO.Path]::GetFileName($archiveFile)) {
        throw 'Invalid archive checksum companion.'
    }
    if ((Get-FileHash -LiteralPath $archiveFile -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Matches[1]) {
        throw 'Archive checksum mismatch.'
    }
    $scratch = Join-Path ([IO.Path]::GetTempPath()) ('tensorfold-smoke-' + [Guid]::NewGuid().ToString('N'))
    [void][IO.Directory]::CreateDirectory($scratch)
    $stageRoot = Join-Path $scratch $name
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $zip = [IO.Compression.ZipFile]::OpenRead($archiveFile)
    try {
        $total = [long]0
        if ($zip.Entries.Count -gt 100000) { throw 'Too many archive entries.' }
        foreach ($entry in $zip.Entries) {
            $path = $entry.FullName
            $parts = $path.TrimEnd('/').Split('/')
            if (-not $path.StartsWith("$name/", [StringComparison]::Ordinal) -or
                $path.Contains('\') -or $path.Contains(':') -or $path.Contains([char]0) -or
                @($parts | Where-Object { $_ -in @('', '.', '..') }).Count -ne 0) { throw 'Unsafe archive path.' }
            if (-not $seen.Add($path.TrimEnd('/'))) { throw 'Duplicate Windows archive path.' }
            $kind = ($entry.ExternalAttributes -shr 16) -band 0xf000
            if ($kind -notin @(0, 0x4000, 0x8000)) { throw 'Nonregular archive entry.' }
            $total += $entry.Length
            if ($entry.Length -gt 1GB -or $total -gt 4GB) { throw 'Oversized archive entry set.' }
            $destination = Join-Path $scratch $path
            if ($path.EndsWith('/')) { [void][IO.Directory]::CreateDirectory($destination); continue }
            [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))
            $source = $entry.Open()
            $target = [IO.File]::Open($destination, [IO.FileMode]::CreateNew)
            try {
                $buffer = [byte[]]::new(1MB)
                $written = [long]0
                while (($count = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $written += $count
                    if ($written -gt $entry.Length) { throw 'Archive payload exceeds its declared size.' }
                    $target.Write($buffer, 0, $count)
                }
                if ($written -ne $entry.Length) { throw 'Truncated archive payload.' }
            } finally { $target.Dispose(); $source.Dispose() }
        }
    } finally { $zip.Dispose() }
    $listed = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in [IO.File]::ReadAllLines((Join-Path $stageRoot 'SHA256SUMS'))) {
        if ($line -notmatch '^([0-9a-f]{64})  (?:\./)?(.+)$') { throw 'Malformed file checksum inventory.' }
        $hash = $Matches[1]; $relative = $Matches[2]
        if (-not $seen.Contains("$name/$relative") -or -not $listed.Add($relative)) { throw 'Invalid checksum member.' }
        if ((Get-FileHash -LiteralPath (Join-Path $stageRoot $relative)).Hash.ToLowerInvariant() -cne $hash) {
            throw 'Packaged file checksum mismatch.'
        }
    }
    foreach ($file in Get-ChildItem -LiteralPath $stageRoot -File -Recurse) {
        $relative = [IO.Path]::GetRelativePath($stageRoot, $file.FullName).Replace('\', '/')
        if ($relative -ne 'SHA256SUMS' -and -not $listed.Contains($relative)) { throw 'Unhashed packaged file.' }
    }
    foreach ($required in @('bin/tensorfold.exe', 'LICENSE', 'NOTICE', 'VERSION', 'RUNTIME.md',
                             'LICENSES/Apache-2.0.txt', 'THIRD_PARTY_NOTICES.md', 'SOURCE.json')) {
        if (-not $listed.Contains($required)) { throw 'Required serving-package file is missing.' }
    }
    if ([IO.File]::ReadAllText((Join-Path $stageRoot 'VERSION')).Trim() -cne $ExpectedVersion) { throw 'VERSION mismatch.' }
    $sourceRecord = ConvertFrom-Json -AsHashtable ([IO.File]::ReadAllText((Join-Path $stageRoot 'SOURCE.json')))
    $binary = Join-Path $stageRoot 'bin/tensorfold.exe'
    if ($sourceRecord['schema'] -ne 1 -or $sourceRecord['target'] -cne 'windows-x86_64' -or
        (-not $hostOnly -and ($sourceRecord['source_dirty'] -ne $false -or $sourceRecord['host_only'] -eq $true)) -or
        $sourceRecord['binary_sha256'] -cne (Get-FileHash -LiteralPath $binary).Hash.ToLowerInvariant()) {
        throw 'Source manifest does not bind this executable.'
    }
    if ($hostOnly) {
        if (-not $listed.Contains('HOST-ONLY')) { throw 'Host-only marker is missing.' }
    } else {
        foreach ($required in @('WINDOWS-RUNTIMES.json', 'WINDOWS-RUNTIME-NOTICES.txt')) {
            if (-not $listed.Contains($required)) { throw 'Runtime provenance and notices are required.' }
        }
    }
    return @{ Root = $stageRoot; Scratch = $scratch; HostOnly = $hostOnly }
}

function Invoke-NativeProbe {
    param([string]$Binary, [string[]]$Arguments, [string]$Scratch)
    $start = [Diagnostics.ProcessStartInfo]::new($Binary)
    $start.UseShellExecute = $false; $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
    $start.WorkingDirectory = Join-Path $Scratch 'empty'
    foreach ($argument in $Arguments) { $start.ArgumentList.Add($argument) }
    $start.Environment.Clear()
    foreach ($key in @('SystemRoot', 'WINDIR')) { $start.Environment[$key] = $env:SystemRoot }
    $start.Environment['PATH'] = Join-Path $env:SystemRoot 'System32'
    $start.Environment['USERPROFILE'] = Join-Path $Scratch 'home'
    $start.Environment['HOME'] = Join-Path $Scratch 'home'
    $start.Environment['TEMP'] = $Scratch; $start.Environment['TMP'] = $Scratch
    $process = [Diagnostics.Process]::Start($start)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync(); $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(30000)) { $process.Kill($true); throw 'Native CPU probe timed out.' }
        if ($process.ExitCode -ne 0) { throw 'Native CPU probe failed.' }
        return $stdout.Result
    } finally { $process.Dispose() }
}

function Invoke-WindowsSmoke {
    param([string]$ArchivePath, [string]$ExpectedVersion, [switch]$HostOnlyAllowed)
    if (-not $IsWindows) { throw 'Executable smoke runs on Windows only; fixture checks can run on any PowerShell host.' }
    $package = Expand-VerifiedWindowsArchive $ArchivePath $ExpectedVersion -HostOnlyAllowed:$HostOnlyAllowed
    foreach ($name in @('empty', 'home')) { [void][IO.Directory]::CreateDirectory((Join-Path $package.Scratch $name)) }
    $binary = Join-Path $package.Root 'bin/tensorfold.exe'
    $versionText = Invoke-NativeProbe $binary @('--version') $package.Scratch
    $helpText = Invoke-NativeProbe $binary @('--help') $package.Scratch
    $serveText = Invoke-NativeProbe $binary @('serve', '--help') $package.Scratch
    $capsText = Invoke-NativeProbe $binary @('capabilities', '--json') $package.Scratch
    Assert-NativeProbe $versionText $helpText $serveText $capsText $ExpectedVersion -HostOnly:$package.HostOnly
    [IO.File]::WriteAllText((Join-Path $package.Scratch 'capabilities.json'), $capsText)
    Write-Output "PASS Windows native version/serve/capabilities from empty cwd; receipt $($package.Scratch)"
}

if ($MyInvocation.InvocationName -ne '.') { Invoke-WindowsSmoke $Archive $Version -HostOnlyAllowed:$AllowHostOnly }
