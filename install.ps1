$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
  & py -3.11 scripts\bootstrap.py @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  & python scripts\bootstrap.py @args
} else {
  throw "Python 3.11+ was not found."
}
exit $LASTEXITCODE
