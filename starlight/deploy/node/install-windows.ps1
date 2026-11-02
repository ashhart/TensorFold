#Requires -RunAsAdministrator
# Installs Starlight Node as a Windows service, enrols it with Starlight and opens its gateway port.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $Server,
    [Parameter(Mandatory = $true)] [string] $Token,
    [Parameter(Mandatory = $true)] [string] $CASha256,
    [Parameter(Mandatory = $true)] [string] $Engine,
    [string] $Binary = ".\starlight-node.exe",
    [string] $Name = $env:COMPUTERNAME.ToLower(),
    [string] $Advertise = "",
    [int] $Port = 7443,
    [string] $InstallDir = "$env:ProgramFiles\TensorFold\Starlight Node",
    [string] $StateDir = "$env:ProgramData\TensorFold\StarlightNode"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Binary)) { throw "starlight-node.exe not found at $Binary" }
if (-not (Test-Path $Engine)) { throw "TensorFold engine not found at $Engine" }
if ($CASha256 -notmatch '^[0-9a-fA-F:]{64,95}$') { throw "-CASha256 must be the SHA-256 the Starlight console prints" }

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Force $Binary (Join-Path $InstallDir "starlight-node.exe")
$exe = Join-Path $InstallDir "starlight-node.exe"

if (-not $Advertise) { $Advertise = "$([System.Net.Dns]::GetHostEntry('').HostName):$Port" }

& $exe enroll --state-dir $StateDir --server $Server --token $Token --ca-sha256 $CASha256 --name $Name --advertise $Advertise --engine $Engine
if ($LASTEXITCODE -ne 0) { throw "enrolment failed" }

& $exe service install --state-dir $StateDir -- --listen ":$Port" --advertise $Advertise --engine $Engine --prevent-sleep
if ($LASTEXITCODE -ne 0) { throw "service installation failed" }

$rule = "TensorFold Starlight Node $Port"
if (-not (Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $rule -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Program $exe | Out-Null
}

& $exe service status
Write-Host "Starlight Node installed. Approve it in the Starlight console if it shows as pending."
