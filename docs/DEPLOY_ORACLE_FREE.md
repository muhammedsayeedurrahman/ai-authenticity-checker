# Deploying the backend on Oracle Cloud's Always Free tier

Genuinely $0/forever hosting for the FastAPI backend, using the existing
`Dockerfile` unchanged. Two things to know going in:

- Oracle requires a card on file for identity verification at signup.
  Always Free resources are not charged, but the card requirement itself
  is real — decide if that's acceptable before starting.
- The Always Free Ampere A1 shape is **ARM64** (aarch64), and as of
  June 2026 Oracle halved the allocation to **2 OCPU / 12GB RAM**
  (previously 4 OCPU / 24GB). This is still enough to run the app, but
  it's tighter than it used to be.

## 1. Create your Oracle Cloud account (you do this)

Go to https://www.oracle.com/cloud/free/ and sign up. This is an account
creation + card-verification step — not something to hand off, do it
yourself.

## 2. Create the Always Free VM instance (you do this, in the OCI console)

1. Console → **Compute → Instances → Create Instance**
2. **Image**: Canonical Ubuntu 22.04 (aarch64/ARM build)
3. **Shape**: click "Change shape" → Ampere → `VM.Standard.A1.Flex` →
   set 2 OCPU / 12 GB (the Always Free allocation)
4. **Networking**: use the default VCN, or create one — just make sure
   "Assign a public IPv4 address" is checked
5. **SSH keys**: generate a new key pair here (or upload your own public
   key) and **save the private key file** — you'll need it to connect
6. Create the instance, note its **public IP** once it's running

## 3. Open the port (OCI console)

By default only SSH (22) is open. Add an ingress rule:

Console → your instance → **Subnet** → **Security Lists** → default
security list → **Add Ingress Rules**:
- Source CIDR: `0.0.0.0/0`
- Destination Port Range: `80,443` (for the reverse proxy in step 7 —
  don't bother opening 7861 directly, see step 7 for why)

Ubuntu's own firewall (`ufw`) is disabled by default on the OCI image,
but if you've enabled it, also `sudo ufw allow 80,443/tcp`.

## 4. SSH in and install Docker

```bash
ssh -i /path/to/your-private-key.pem ubuntu@<VM_PUBLIC_IP>

# Docker + compose plugin (Ubuntu 22.04, arm64)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker compose version   # sanity check
```

## 5. Get the code and model weights onto the VM

```bash
git clone https://github.com/ab-rar-6024/ai-authenticity-checker.git
cd ai-authenticity-checker
```

`models/` (674MB of `.pth`/`.joblib` weights) is gitignored — `git clone`
won't bring them. Copy them from your own machine instead, **from a
second terminal on your PC** (not the VM):

```bash
scp -i /path/to/your-private-key.pem -r \
  "D:\Microsoft VS Code\ai-authenticity-checker\models" \
  ubuntu@<VM_PUBLIC_IP>:~/ai-authenticity-checker/models
```

This will take a while over a home connection (674MB).

## 6. Configure environment

```bash
cp .env.example .env
nano .env
```

Set at minimum:
```
CORS_ORIGINS=https://your-app.vercel.app
```
(Leave `DATABASE_URL` unset to use the bundled Postgres from
docker-compose, or blank for SQLite if you edit that out — the compose
file already wires up Postgres for you.)

## 7. Reverse proxy + HTTPS (required — see the mixed-content note above)

The included `docker-compose.yml` exposes the app on port 7861 over
plain HTTP. Your Vercel frontend is HTTPS-only, so the browser will
block calls to a plain-HTTP backend. You need a domain pointed at the
VM's IP and a TLS cert. **Caddy** does this in about 4 lines with
automatic Let's Encrypt certs — install it alongside the app:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```

Point a domain/subdomain's A record at `<VM_PUBLIC_IP>` first (any
registrar — even a free one like a DuckDNS subdomain works), then:

```bash
sudo tee /etc/caddy/Caddyfile <<'EOF'
api.yourdomain.com {
    reverse_proxy localhost:7861
}
EOF
sudo systemctl restart caddy
```

Caddy handles the cert automatically once DNS resolves. Update
`CORS_ORIGINS` and any frontend `VITE_API_URL` to use
`https://api.yourdomain.com`, not the bare IP.

## 8. Before running compose: drop the GPU reservation

`docker-compose.yml` requests an nvidia GPU device by default. This VM
has no GPU (and is ARM, so nvidia's toolkit doesn't apply anyway) —
`docker compose up` will fail with that block in place. Edit
`docker-compose.yml` and delete this section from the `proofyx` service
before building:

```yaml
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

(Keep the `memory: 8G` limit, or lower it — you only have 12GB total on
this shape, shared with Postgres.)

## 9. Build and run

```bash
docker compose up --build -d
docker compose logs -f proofyx   # watch model loading; first build/boot is slow
```

## 10. Verify

```bash
curl http://localhost:7861/api/v1/health
curl https://api.yourdomain.com/api/v1/health   # once Caddy + DNS are live
```

Then set your Vercel project's `VITE_API_URL` env var to
`https://api.yourdomain.com` and redeploy the frontend.
