# ReasonKit Web MCP Sidecar - systemd Deployment

Production deployment configuration for Debian 13 (Trixie).

## Files

| File | Description |
|------|-------------|
| `reasonkit-web.service` | systemd unit file with security hardening |
| `reasonkit-web.env.example` | Environment configuration template |
| `install.sh` | Automated installation script |

## Quick Start

```bash
# Build the binary
cargo build --release

# Install as root
sudo ./install.sh --binary ../target/release/reasonkit-web
```

## Manual Installation

```bash
# 1. Create service user
sudo useradd --system --shell /usr/sbin/nologin --home-dir /var/lib/reasonkit reasonkit

# 2. Create directories
sudo mkdir -p /opt/reasonkit/bin /etc/reasonkit /var/lib/reasonkit /var/log/reasonkit
sudo chown reasonkit:reasonkit /var/lib/reasonkit /var/log/reasonkit

# 3. Install binary
sudo install -m 0755 target/release/reasonkit-web /opt/reasonkit/bin/

# 4. Install service
sudo cp reasonkit-web.service /etc/systemd/system/
sudo cp reasonkit-web.env.example /etc/reasonkit/reasonkit-web.env
sudo chmod 640 /etc/reasonkit/reasonkit-web.env

# 5. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now reasonkit-web.service
```

## Security Features

The service unit implements defense-in-depth security:

- **Filesystem Isolation**: `ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp=yes`
- **Capability Dropping**: `CapabilityBoundingSet=` (empty = no capabilities)
- **Privilege Escalation Prevention**: `NoNewPrivileges=yes`
- **Namespace Isolation**: `PrivateUsers=yes`, `ProtectHostname=yes`
- **System Call Filtering**: Allowlist-based syscall filter
- **Memory Protection**: `MemoryDenyWriteExecute=yes`
- **Network Restrictions**: Only localhost and private networks

## Resource Limits

| Resource | Limit |
|----------|-------|
| Memory | 512M max, 448M high watermark |
| CPU | 100% quota |
| Tasks | 100 max |
| File descriptors | 65536 |

## Watchdog

The service uses systemd's watchdog feature:

- `WatchdogSec=30`: Service must notify within 30 seconds
- Implement `sd_notify` in the binary with `WATCHDOG=1` messages
- Failure to notify triggers automatic restart

## Logs

```bash
# View logs
journalctl -u reasonkit-web.service -f

# View logs since boot
journalctl -u reasonkit-web.service -b

# View last 100 lines
journalctl -u reasonkit-web.service -n 100
```

## Management

```bash
# Status
sudo systemctl status reasonkit-web

# Restart
sudo systemctl restart reasonkit-web

# Reload (SIGHUP)
sudo systemctl reload reasonkit-web

# Stop
sudo systemctl stop reasonkit-web
```

## Upgrade

```bash
cargo build --release
sudo ./install.sh --upgrade --binary ../target/release/reasonkit-web
```

## Uninstall

```bash
sudo ./install.sh --uninstall
```

## Configuration

Edit `/etc/reasonkit/reasonkit-web.env` to customize:

```bash
# Essential settings
RUST_LOG=info
REASONKIT_WEB_HOST=127.0.0.1
REASONKIT_WEB_PORT=3847

# Enable TLS for production
REASONKIT_TLS_ENABLED=true
REASONKIT_TLS_CERT=/etc/reasonkit/tls/cert.pem
REASONKIT_TLS_KEY=/etc/reasonkit/tls/key.pem
```

After editing, restart the service:

```bash
sudo systemctl restart reasonkit-web
```
