#requires -Version 7.0
param([string]$FixtureArchive)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'smoke-windows.ps1')
$fixtureVersion = '0.6.5'
$caps = @{ schema = 1; engine = 'zig'; version = $fixtureVersion; backends = @('cuda');
    families = @{ qwen3_5 = @('nvfp4') }; serve = @{ flags = @{ context = @{}; parallel = @{} } };
    optional_libraries = @(@{ soname = 'tfgrammar.dll'; found = $true }) }
$good = @{
    VersionText = "tensorfold-native $fixtureVersion"; HelpText = 'serve MODEL';
    ServeText = 'usage: tensorfold serve MODEL'; CapabilitiesText = ($caps | ConvertTo-Json -Depth 8);
    ExpectedVersion = $fixtureVersion
}
Assert-NativeProbe @good
function Refuses([hashtable]$Candidate) {
    $refused = $false
    try { Assert-NativeProbe @Candidate } catch { $refused = $true }
    if (-not $refused) { throw 'A wrong executable or capabilities fixture passed the Windows smoke.' }
}
$bad = $good.Clone(); $bad.VersionText = "tensorfold $fixtureVersion"; Refuses $bad
$bad = $good.Clone(); $bad.HelpText = 'info REPO, pull REPO, scan DIRECTORY'; Refuses $bad
$bad = $good.Clone(); $bad.ServeText = 'unknown command serve'; Refuses $bad
$bad = $good.Clone(); $bad.CapabilitiesText = '{}'; Refuses $bad
$caps.engine = 'checkpoint'; $bad = $good.Clone(); $bad.CapabilitiesText = ($caps | ConvertTo-Json -Depth 8); Refuses $bad
$caps.engine = 'zig'; $caps.version = '0.0.0'
$bad = $good.Clone(); $bad.CapabilitiesText = ($caps | ConvertTo-Json -Depth 8); Refuses $bad
$caps.version = $fixtureVersion; $caps.backends = @('metal')
$bad = $good.Clone(); $bad.CapabilitiesText = ($caps | ConvertTo-Json -Depth 8); Refuses $bad
$caps.backends = @('cuda'); $caps.serve.flags.Remove('parallel')
$bad = $good.Clone(); $bad.CapabilitiesText = ($caps | ConvertTo-Json -Depth 8); Refuses $bad
$caps.serve.flags.parallel = @{}; $caps.optional_libraries[0].found = $false
$bad = $good.Clone(); $bad.CapabilitiesText = ($caps | ConvertTo-Json -Depth 8); Refuses $bad
Write-Output 'PASS native serving identity, legacy version protocol and nine wrong-CLI/capability refusal controls'

if ($FixtureArchive) {
    $package = Expand-VerifiedWindowsArchive $FixtureArchive $fixtureVersion
    if (-not [IO.File]::Exists((Join-Path $package.Root 'bin/tensorfold.exe'))) { throw 'Canonical filename missing.' }
    $prefix = "tensorfold-$fixtureVersion-windows-x86_64"
    function MutatedArchiveRefuses([scriptblock]$Mutation, [string]$Expected) {
        $directory = Join-Path ([IO.Path]::GetTempPath()) ('tensorfold-smoke-control-' + [Guid]::NewGuid().ToString('N'))
        [void][IO.Directory]::CreateDirectory($directory)
        $archive = Join-Path $directory "$prefix.zip"
        [IO.File]::Copy($FixtureArchive, $archive)
        $zip = [IO.Compression.ZipFile]::Open($archive, [IO.Compression.ZipArchiveMode]::Update)
        try { & $Mutation $zip } finally { $zip.Dispose() }
        $hash = (Get-FileHash -LiteralPath $archive).Hash.ToLowerInvariant()
        [IO.File]::WriteAllText("$archive.sha256", "$hash  $prefix.zip`n")
        $failure = ''
        try { $null = Expand-VerifiedWindowsArchive $archive $fixtureVersion } catch { $failure = $_.Exception.Message }
        if ($failure -notmatch $Expected) { throw "Archive control missed its gate: $Expected" }
    }
    MutatedArchiveRefuses { param($zip) $null = $zip.CreateEntry("$prefix/../escape.txt") } 'Unsafe archive path'
    MutatedArchiveRefuses { param($zip) $null = $zip.CreateEntry("$prefix/license") } 'Duplicate Windows archive path'
    MutatedArchiveRefuses {
        param($zip)
        $entry = $zip.CreateEntry("$prefix/link")
        $entry.ExternalAttributes = -1577123840
    } 'Nonregular archive entry'
    MutatedArchiveRefuses {
        param($zip)
        $zip.GetEntry("$prefix/VERSION").Delete()
        $entry = $zip.CreateEntry("$prefix/VERSION"); $stream = $entry.Open()
        try { $bytes = [Text.Encoding]::UTF8.GetBytes('changed'); $stream.Write($bytes, 0, $bytes.Length) }
        finally { $stream.Dispose() }
    } 'Packaged file checksum mismatch'
    Write-Output 'PASS Windows archive extraction/checksum controls without executing the synthetic PE or DLL'
}
