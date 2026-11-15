# Read-only check: is `hypervisorlaunchtype` actually set in the boot entry?
#
# Needs elevation only because `bcdedit /enum` refuses to read the store
# otherwise. Changes nothing -- safe to run any time.
#
# ## Why this exists
#
# `bcdedit /set hypervisorlaunchtype auto` returned 0 in an elevated window, but
# the *registry* value `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\
# HypervisorLaunchType` was still absent afterwards. That is expected and does
# NOT mean the fix failed:
#
#   hypervisorlaunchtype  lives in the **BCD store** (the boot configuration),
#                         not in that registry key. `bcdedit` writes there.
#   HypervisorLaunchType  is an optional policy value in that key; Windows reads
#                         the BCD one. Some docs conflate the two.
#
# So the only honest check is `bcdedit /enum {current}` from an elevated context.

$ErrorActionPreference = 'Continue'
$log = Join-Path $PSScriptRoot '..\reports\hypervisor_bcd_check.log'
$out = New-Object System.Collections.Generic.List[string]

function Say([string]$t) { $out.Add($t); Write-Host $t }

Say "=== BCD check: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Say "elevated: $elevated"
if (-not $elevated) {
    Say 'NOT elevated -- bcdedit will refuse to read the store. Exiting.'
    $out | Set-Content -Path $log -Encoding UTF8
    exit 1
}

Say ''
Say '--- bcdedit /enum {current} | hypervisor lines ---'
$enum = & bcdedit /enum "{current}" 2>&1
$enum | Out-String | Write-Host
$hits = $enum | Select-String -Pattern 'hypervisor'
if ($hits) {
    Say 'MATCHES:'
    $hits | ForEach-Object { Say ("  " + $_.Line.Trim()) }
} else {
    Say 'no hypervisor* line in the current boot entry'
}

Say ''
Say '--- verdict ---'
$launch = ($enum | Select-String -Pattern 'hypervisorlaunchtype\s+(\S+)').Matches
if ($launch) {
    $val = $launch[0].Groups[1].Value
    if ($val -ieq 'Auto') {
        Say "hypervisorlaunchtype = $val  -> SET. Reboot and WSL2 should start."
    } else {
        Say "hypervisorlaunchtype = $val  -> set, but not Auto. Hypervisor will not load at boot."
    }
} else {
    Say 'hypervisorlaunchtype is NOT in the boot entry -> the bcdedit write did not stick.'
}

Say ''
Say '--- optional feature state ---'
foreach ($f in 'VirtualMachinePlatform', 'HypervisorPlatform') {
    $s = (Get-CimInstance Win32_OptionalFeature -Filter "Name='$f'" -ErrorAction SilentlyContinue).InstallState
    Say ("{0,-24} = {1}" -f $f, $(switch ($s) { 1 {'Enabled(1)'} 2 {'Disabled(2)'} 3 {'Absent(3)'} default {"unknown($s)"} }))
}
Say ''
Say "HypervisorPresent = $((Get-CimInstance Win32_ComputerSystem).HypervisorPresent)  (expected False until reboot)"

$out | Set-Content -Path $log -Encoding UTF8
Write-Host ''
Write-Host "log: $log"
Write-Host 'Press Enter to close.'
Read-Host | Out-Null
