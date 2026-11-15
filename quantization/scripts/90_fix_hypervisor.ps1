# Fix the WSL2 hypervisor blocker on this machine.
#
# Run elevated (the caller does `Start-Process -Verb RunAs`):
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/90_fix_hypervisor.ps1
#
# ## What is actually wrong (measured 2026-10-23)
#
#   CPU VirtualizationFirmwareEnabled = True      <- firmware is fine, do NOT touch BIOS
#   CPU VMMonitorModeExtensions       = True
#   HypervisorPresent                 = False     <- the hypervisor is not running
#   HKLM\...\Session Manager\HypervisorLaunchType = ABSENT (defaults to Off)
#   VirtualMachinePlatform            = Installed (InstallState 1)
#   HypervisorPlatform                = Disabled  (InstallState 2)
#   wsl -e  ->  HCS_E_HYPERV_NOT_INSTALLED
#
# So the firmware is not the problem and `wsl --install` is not needed. Two things
# are: the boot entry never asks Windows to start the hypervisor, and the
# `HypervisorPlatform` optional feature (the WinHv API WSL2's VM host uses) is off.
#
# ## Why this needs a reboot
#
# `hypervisorlaunchtype` is read by the boot loader. There is no way to load the
# hypervisor into a running kernel, so `/norestart` here leaves the change staged
# and the machine keeps working (WSL2 still broken) until it reboots.
#
# ## Side effects worth knowing before you reboot
#
# With the hypervisor loaded, any bare-metal virtualisation (VMware Workstation,
# VirtualBox in non-Hyper-V mode) runs on top of Hyper-V instead of the CPU.
# Docker Desktop is already installed here and uses the docker-desktop WSL2
# distribution, so it wants the hypervisor anyway.
# Reversible: `bcdedit /set hypervisorlaunchtype off` + reboot.

$ErrorActionPreference = 'Continue'
$log = Join-Path $PSScriptRoot '..\reports\hypervisor_fix.log'
$lines = New-Object System.Collections.Generic.List[string]

function Say([string]$text) {
    $lines.Add($text)
    Write-Host $text
}

Say "=== hypervisor fix: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Say "elevated: $((([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)))"

# --- before ---------------------------------------------------------------
Say ''
Say '--- state before ---'
$cs = Get-CimInstance Win32_ComputerSystem
Say "HypervisorPresent          = $($cs.HypervisorPresent)"
$launch = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' `
           -Name HypervisorLaunchType -ErrorAction SilentlyContinue).HypervisorLaunchType
Say "HypervisorLaunchType       = $(if ($null -eq $launch) { 'ABSENT' } else { $launch })"
foreach ($f in 'VirtualMachinePlatform', 'HypervisorPlatform') {
    $s = (Get-CimInstance Win32_OptionalFeature -Filter "Name='$f'" -ErrorAction SilentlyContinue).InstallState
    # 1 = Enabled, 2 = Disabled, 3 = Absent
    Say ("{0,-24} = {1}" -f $f, $(switch ($s) { 1 {'Enabled(1)'} 2 {'Disabled(2)'} 3 {'Absent(3)'} default {"unknown($s)"} }))
}

# --- 1. boot entry: ask for the hypervisor ---------------------------------
Say ''
Say '--- bcdedit /set hypervisorlaunchtype auto ---'
$out = & bcdedit /set hypervisorlaunchtype auto 2>&1
Say ($out | Out-String).Trim()
Say "exit code: $LASTEXITCODE"

# --- 2. optional feature ----------------------------------------------------
Say ''
Say '--- enable HypervisorPlatform (no restart) ---'
$out = & dism.exe /online /enable-feature /featurename:HypervisorPlatform /all /norestart 2>&1
Say ($out | Out-String).Trim()
Say "exit code: $LASTEXITCODE"

# VirtualMachinePlatform is already enabled, but re-assert it: a machine where it
# got toggled off would fail in a confusingly different way.
Say ''
Say '--- ensure VirtualMachinePlatform is enabled (no restart) ---'
$out = & dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart 2>&1
Say ($out | Out-String).Trim()
Say "exit code: $LASTEXITCODE"

# --- after ----------------------------------------------------------------
Say ''
Say '--- state after (hypervisor is expected to stay False until reboot) ---'
$launch = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' `
           -Name HypervisorLaunchType -ErrorAction SilentlyContinue).HypervisorLaunchType
Say "HypervisorLaunchType       = $(if ($null -eq $launch) { 'ABSENT' } else { $launch })"
foreach ($f in 'VirtualMachinePlatform', 'HypervisorPlatform') {
    $s = (Get-CimInstance Win32_OptionalFeature -Filter "Name='$f'" -ErrorAction SilentlyContinue).InstallState
    Say ("{0,-24} = {1}" -f $f, $(switch ($s) { 1 {'Enabled(1)'} 2 {'Disabled(2)'} 3 {'Absent(3)'} default {"unknown($s)"} }))
}
Say ''
Say 'DONE. Reboot is required before WSL2 works.'
Say 'Verify after reboot:  wsl -d Ubuntu -e echo ok'

$lines | Set-Content -Path $log -Encoding UTF8
Write-Host ''
Write-Host "log: $log"
Write-Host 'Press Enter to close this window.'
Read-Host | Out-Null
