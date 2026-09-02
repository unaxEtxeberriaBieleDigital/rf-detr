<#
.SYNOPSIS
    Builds the RF-DETR Visualizer as a self-contained, portable Windows application.

.DESCRIPTION
    Produces a distributable directory containing only compiled artifacts, so no Python or
    TypeScript source is shipped:

      * the FastAPI backend is frozen with PyInstaller (sources become bytecode inside the archive),
      * the React frontend is embedded into the Tauri executable by `npm run build`.

    The result is a portable folder that runs without installation. Installers are intentionally
    not produced: the CUDA-enabled PyTorch payload is far larger than the practical limits of the
    WiX/NSIS bundlers.

.PARAMETER SkipBackend
    Reuse the PyInstaller bundle already staged under src-tauri\resources. Saves several minutes
    when only the frontend or the Rust shell changed.

.EXAMPLE
    .\build_visualizer.ps1

.EXAMPLE
    .\build_visualizer.ps1 -SkipBackend
#>

[CmdletBinding()]
param(
    [switch]$SkipBackend
)

$ErrorActionPreference = "Stop"

$AppName = "DataCleaner"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$FrontendDir = Join-Path $ScriptDir "frontend"
$BuildDir = Join-Path $ScriptDir "build"
$TauriResourcesDir = Join-Path $FrontendDir "src-tauri\resources"
$BackendBundleDir = Join-Path $TauriResourcesDir "visualizer-backend"
$PyInstallerWorkDir = Join-Path $BuildDir "pyinstaller"
$DistDir = Join-Path $ScriptDir "dist\$AppName"

function Get-DirectorySizeMb {
    param([Parameter(Mandatory)][string]$Path)

    $files = Get-ChildItem -LiteralPath $Path -Recurse -File -Force
    if (-not $files) { return 0 }
    return [math]::Round(($files | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
}

<#
    PyInstaller collects the CUDA runtime twice: once next to the executable and once inside
    torch\lib. Only the torch\lib copy is ever loaded, because torch registers that directory with
    os.add_dll_directory() before importing its extension modules. Dropping the unused copies keeps
    GPU support intact while removing ~2.8 GB from the bundle.
#>
function Remove-DuplicateTorchLibraries {
    param([Parameter(Mandatory)][string]$InternalDir)

    $torchLibDir = Join-Path $InternalDir "torch\lib"
    if (-not (Test-Path -LiteralPath $torchLibDir)) {
        Write-Warning "torch\lib not found; skipping CUDA de-duplication."
        return
    }

    $torchLibFiles = @{}
    foreach ($file in Get-ChildItem -LiteralPath $torchLibDir -File -Force) {
        $torchLibFiles[$file.Name] = $file.Length
    }

    $duplicates = Get-ChildItem -LiteralPath $InternalDir -File -Force |
        Where-Object { $torchLibFiles.ContainsKey($_.Name) -and $torchLibFiles[$_.Name] -eq $_.Length }

    if (-not $duplicates) {
        Write-Host "No duplicated CUDA libraries found."
        return
    }

    $reclaimedMb = [math]::Round(($duplicates | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    $duplicates | Remove-Item -Force
    Write-Host "Removed $($duplicates.Count) duplicated CUDA libraries ($reclaimedMb MB reclaimed)."
}

Push-Location $RootDir
try {
    if ($SkipBackend) {
        if (-not (Test-Path -LiteralPath (Join-Path $BackendBundleDir "visualizer-backend.exe"))) {
            throw "-SkipBackend was requested but no staged backend exists. Run without -SkipBackend first."
        }
        Write-Host "== Reusing the staged backend bundle =="
    }
    else {
        Write-Host "== Freezing the backend with PyInstaller =="
        if (Test-Path -LiteralPath $BackendBundleDir) {
            Remove-Item -Recurse -Force -LiteralPath $BackendBundleDir
        }

        uv run --no-sync --extra visualizer --with pyinstaller pyinstaller `
            --noconfirm `
            --clean `
            --onedir `
            --name visualizer-backend `
            --paths $RootDir `
            --collect-submodules visualizer.backend `
            --collect-all rfdetr `
            --collect-all sklearn `
            --collect-all supervision `
            --collect-all torchvision `
            --collect-all transformers `
            --collect-all umap `
            --distpath $TauriResourcesDir `
            --workpath $PyInstallerWorkDir `
            --specpath $PyInstallerWorkDir `
            (Join-Path $ScriptDir "backend\launcher.py")

        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE."
        }

        New-Item -ItemType File -Path (Join-Path $BackendBundleDir ".gitkeep") -Force | Out-Null

        Write-Host "== Removing duplicated CUDA libraries =="
        Remove-DuplicateTorchLibraries -InternalDir (Join-Path $BackendBundleDir "_internal")
    }

    Write-Host "== Building the Tauri application =="
    Push-Location $FrontendDir
    try {
        # --no-bundle stops at the executable: the installers cannot handle a multi-gigabyte payload,
        # and the portable layout below is what gets distributed.
        npm run tauri build -- --no-bundle
        if ($LASTEXITCODE -ne 0) {
            throw "Tauri build failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "== Assembling the portable application =="
    $builtExecutable = Join-Path $FrontendDir "src-tauri\target\release\$AppName.exe"
    if (-not (Test-Path -LiteralPath $builtExecutable)) {
        throw "The Tauri executable was not found at '$builtExecutable'."
    }

    if (Test-Path -LiteralPath $DistDir) {
        Remove-Item -Recurse -Force -LiteralPath $DistDir
    }
    New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

    Copy-Item -LiteralPath $builtExecutable -Destination $DistDir

    # Tauri resolves resource_dir() to the executable's own directory in a portable layout, and
    # src-tauri\src\lib.rs looks for the backend under <resource_dir>\backend.
    $DistBackendDir = Join-Path $DistDir "backend"
    robocopy $BackendBundleDir $DistBackendDir /MIR /NFL /NDL /NJH /NJS /NP /XF ".gitkeep" | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "Copying the backend bundle failed with robocopy exit code $LASTEXITCODE."
    }
    $global:LASTEXITCODE = 0

    Write-Host ""
    Write-Host "Portable application ready: $DistDir"
    Write-Host ("Total size: {0:N1} MB" -f (Get-DirectorySizeMb -Path $DistDir))
    Write-Host "Distribute the whole '$AppName' folder and launch $AppName.exe."
}
finally {
    Pop-Location
}
