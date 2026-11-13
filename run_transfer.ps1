$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:EWS_PLAN_COMMIT = '5b4e7739a9f86c5f98a43e7d4e166710c111bb49'
$env:EWS_IMPLEMENTATION_COMMIT = (& git -c "safe.directory=$($PSScriptRoot.Replace('\','/'))" rev-parse HEAD).Trim()
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$startedUtc = [DateTime]::UtcNow.ToString('o')
$pythonPath = Join-Path $PSScriptRoot '..\transfer-ews\.venv\Scripts\python.exe'
$runExit = 1
try {
    # Python warnings on stderr must not terminate the PowerShell supervisor.
    $ErrorActionPreference = 'Continue'
    & $pythonPath -u run_all.py --config config.yaml --tiers 2,3,4 *> transfer_real.log
    $runExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
}
finally {
    $timer.Stop()
    @{
        exit_code = $runExit
        started_utc = $startedUtc
        finished_utc = [DateTime]::UtcNow.ToString('o')
        elapsed_seconds = $timer.Elapsed.TotalSeconds
        plan_commit = $env:EWS_PLAN_COMMIT
        implementation_commit = $env:EWS_IMPLEMENTATION_COMMIT
        command = 'run_all.py --config config.yaml --tiers 2,3,4'
    } | ConvertTo-Json | Set-Content transfer_execution.json
    Copy-Item -LiteralPath transfer_execution.json,transfer_real.log -Destination outputs\provenance
    if ($runExit -eq 0) {
        Compress-Archive -LiteralPath (Join-Path $PSScriptRoot 'outputs') -DestinationPath (Join-Path $PSScriptRoot 'outputs_all_tiers_real.zip') -Force
    }
}
exit $runExit
