# ReasonKit Web - Installation Scripts

Production installation scripts for Debian 13 (Trixie) and later.

## Quick Start

```bash
# Build the binary first
cargo build --release

# Install (as root)
sudo ./scripts/install.sh

# Configure
sudo ./scripts/configure.sh --start --enable

# Verify installation
./scripts/verify.sh
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `install.sh` | Install binary, create user, setup systemd |
| `configure.sh` | Interactive or automated configuration |
| `uninstall.sh` | Remove installation |
| `verify.sh` | Post-installation verification |
| `reasonkit-web.service` | Systemd service template |

## Installation (install.sh)

Performs complete production installation:

1. Pre-flight checks (Debian 13+, root, dependencies)
2. Creates `reasonkit` system user
3. Creates directory structure
4. Installs binary to `/opt/reasonkit/bin/`
5. Installs Chromium (optional)
6. Configures systemd service
7. Sets up logrotate

### Usage

```bash
sudo ./install.sh [OPTIONS]

Options:
  --binary PATH    Path to pre-built binary
  --prefix PATH    Installation prefix (default: /opt/reasonkit)
  --skip-user      Skip user creation
  --skip-chromium  Skip Chromium installation
  --help           Show help
```

### Examples

```bash
# Standard installation (builds if needed)
sudo ./install.sh

# Use pre-built binary
sudo ./install.sh --binary ./reasonkit-web

# Custom prefix
sudo ./install.sh --prefix /usr/local
```

## Configuration (configure.sh)

Interactive wizard or non-interactive configuration:

### Interactive Mode

```bash
sudo ./configure.sh --start --enable
```

Prompts for:
- Log level
- Chrome/Chromium path
- Headless mode
- MCP timeout
- Worker threads

### Non-Interactive Mode

```bash
sudo RUST_LOG=debug \
     MCP_TIMEOUT_SECS=60 \
     ./configure.sh --non-interactive --start --enable
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (error/warn/info/debug/trace) |
| `CHROME_PATH` | auto | Chrome/Chromium binary path |
| `REASONKIT_HEADLESS` | `true` | Run browser headless |
| `REASONKIT_DISABLE_GPU` | `true` | Disable GPU acceleration |
| `MCP_TIMEOUT_SECS` | `30` | MCP request timeout |
| `TOKIO_WORKER_THREADS` | `4` | Async runtime threads |

## Uninstallation (uninstall.sh)

```bash
# Basic uninstall (preserves config/data)
sudo ./uninstall.sh

# Complete removal
sudo ./uninstall.sh --purge --remove-user --yes
```

### Options

| Option | Description |
|--------|-------------|
| `--purge` | Remove config, data, and logs |
| `--remove-user` | Remove the reasonkit user |
| `--yes` | Skip confirmation prompts |

## Verification (verify.sh)

Validates the installation:

```bash
# Basic check
./verify.sh

# Detailed output
./verify.sh --verbose

# JSON output (for monitoring)
./verify.sh --json
```

### Checks Performed

- Binary installation
- Symlink correctness
- Service user exists
- Directory permissions
- Configuration file
- Systemd service file
- Service status
- Chromium availability
- Memory usage
- Recent errors

## Directory Structure

After installation:

```
/opt/reasonkit/
  bin/
    reasonkit-web         # Binary

/etc/reasonkit/
  reasonkit-web.env       # Configuration

/var/lib/reasonkit/       # Runtime data

/var/log/reasonkit/       # Log files

/etc/systemd/system/
  reasonkit-web.service   # Systemd unit
```

## Service Management

```bash
# Start/stop/restart
sudo systemctl start reasonkit-web
sudo systemctl stop reasonkit-web
sudo systemctl restart reasonkit-web

# Enable on boot
sudo systemctl enable reasonkit-web

# Check status
sudo systemctl status reasonkit-web

# View logs
journalctl -u reasonkit-web -f
```

## Troubleshooting

### Service fails to start

1. Check logs: `journalctl -u reasonkit-web -n 50`
2. Verify Chrome: `chromium --version`
3. Check config: `cat /etc/reasonkit/reasonkit-web.env`
4. Run verify: `./verify.sh --verbose`

### Permission denied errors

```bash
# Fix data directory permissions
sudo chown -R reasonkit:reasonkit /var/lib/reasonkit
sudo chmod 750 /var/lib/reasonkit
```

### High memory usage

Edit `/etc/reasonkit/reasonkit-web.env`:

```bash
TOKIO_WORKER_THREADS=2
```

Or adjust systemd limits in the service file.

## Security Notes

The systemd service includes security hardening:

- `NoNewPrivileges=yes`
- `ProtectSystem=strict`
- `ProtectHome=yes`
- `PrivateDevices=yes`
- Restricted address families
- Restricted namespaces
- Memory limits (2GB default)

## Requirements

- Debian 13 (Trixie) or later
- Root/sudo access
- Chromium or Google Chrome
- Rust toolchain (for building)

## License

Apache License 2.0
