$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
  throw "Cargo is not installed. Install Rust from https://rustup.rs, reopen PowerShell, then rerun this script."
}
Push-Location rust\meteor-engine
cargo build --release
Pop-Location
Write-Host "Rust engine built successfully." -ForegroundColor Green
