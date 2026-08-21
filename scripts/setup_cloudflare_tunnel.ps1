<#
.SYNOPSIS
Set up a named Cloudflare Tunnel that exposes ONLY the capture image path.

.DESCRIPTION
LINE fetches an image message's URL from its own servers, so a picture on this
machine has to be reachable over HTTPS for that fetch. A named tunnel gives a
stable hostname, unlike a quick tunnel whose address changes on every restart.

The ingress rules below route exactly one path prefix to the local server and
answer everything else with 404 before it ever reaches the application. The
server refuses non-loopback requests to every other path anyway -- this is the
second of the two locks, not the only one.

You must run `cloudflared tunnel login` yourself first: it opens a browser and
authorises against your Cloudflare account, which is not something a script
should do on your behalf.

.EXAMPLE
  .\scripts\setup_cloudflare_tunnel.ps1 -Hostname drone.example.com
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Hostname,

    [string]$TunnelName = "telemetry-ocr",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
if (-not (Test-Path -LiteralPath $cloudflared)) {
    $found = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $found) { throw "cloudflared not found. Install it, then run this again." }
    $cloudflared = $found.Source
}

$cfDir = Join-Path $env:USERPROFILE ".cloudflared"
if (-not (Test-Path -LiteralPath (Join-Path $cfDir "cert.pem"))) {
    Write-Host ""
    Write-Host "You are not logged in to Cloudflare yet." -ForegroundColor Yellow
    Write-Host "Run this first, finish it in the browser, then run this script again:"
    Write-Host ""
    Write-Host "    `"$cloudflared`" tunnel login"
    Write-Host ""
    Write-Host "It needs a domain already on your Cloudflare account -- pick that domain"
    Write-Host "in the browser, and use a subdomain of it for -Hostname."
    exit 1
}

# Reuse the tunnel if it exists. Creating a second one with the same name fails,
# and re-running this script after a change should not be a destructive act.
$existing = & $cloudflared tunnel list --output json | ConvertFrom-Json |
    Where-Object { $_.name -eq $TunnelName }
if ($existing) {
    Write-Host "Tunnel '$TunnelName' already exists ($($existing.id))"
    $tunnelId = $existing.id
} else {
    Write-Host "Creating tunnel '$TunnelName'..."
    & $cloudflared tunnel create $TunnelName
    if ($LASTEXITCODE -ne 0) { throw "Could not create the tunnel." }
    $tunnelId = (& $cloudflared tunnel list --output json | ConvertFrom-Json |
        Where-Object { $_.name -eq $TunnelName }).id
}

Write-Host "Pointing $Hostname at the tunnel..."
& $cloudflared tunnel route dns $TunnelName $Hostname
if ($LASTEXITCODE -ne 0) {
    Write-Host "DNS route failed. If the record already exists that is fine." -ForegroundColor Yellow
}

# Ingress is evaluated top to bottom. The catch-all 404 at the end is what makes
# every other path unreachable through the tunnel, including the page itself.
$configPath = Join-Path $cfDir "config.yml"
$credentials = Join-Path $cfDir "$tunnelId.json"
$config = @"
# Written by scripts/setup_cloudflare_tunnel.ps1
#
# One path is routed to the local server: the capture images LINE has to fetch.
# Everything else is answered with 404 here, before it reaches the application.
tunnel: $tunnelId
credentials-file: $credentials

ingress:
  - hostname: $Hostname
    path: ^/public/img/.+
    service: http://127.0.0.1:$Port
  - service: http_status:404
"@
Set-Content -LiteralPath $configPath -Value $config -Encoding utf8
Write-Host "Wrote $configPath"

Write-Host ""
Write-Host "Next:" -ForegroundColor Green
Write-Host "  1. Install the tunnel as a service so it survives a reboot:"
Write-Host "         `"$cloudflared`" service install"
Write-Host "  2. Put this in configs\line.json so captures carry their picture:"
Write-Host "         `"public_base_url`": `"https://$Hostname`""
Write-Host "  3. Check it from outside this machine:"
Write-Host "         https://$Hostname/public/img/anything   -> should be 404, not a page"
Write-Host ""
Write-Host "Only /public/img/<token> is routed. Every other path returns 404 at the"
Write-Host "tunnel, and the server refuses non-loopback requests to them regardless."
