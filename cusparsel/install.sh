#!/usr/bin/env bash
set -euo pipefail

CUSPARSELT_VERSION="0.8.0"
DISTRO="ubuntu2404"
DEB="cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_${CUSPARSELT_VERSION}-1_arm64.deb"

echo "[cuSPARSELt] Downloading local repo package..."
wget -q --show-progress --progress=bar:force:noscroll --no-check-certificate \
  -O "${DEB}" \
  "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/${DEB}"

echo "[cuSPARSELt] Installing local repo..."
sudo dpkg -i "${DEB}"

echo "[cuSPARSELt] Installing GPG key..."
sudo cp /var/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}/cusparselt-local-*-keyring.gpg /usr/share/keyrings/

echo "[cuSPARSELt] Updating APT cache for cuSPARSELt repo..."
sudo apt-get update

# Detect CUDA major if nvcc is available, otherwise default to 13
if command -v nvcc >/dev/null 2>&1; then
  CUDA_MAJOR=$(nvcc --version | awk -F'release ' '/release/{print $2}' | cut -d. -f1)
else
  CUDA_MAJOR="13"
fi
echo "[cuSPARSELt] Using CUDA major ${CUDA_MAJOR}"

# Prefer CUDA-major-suffixed packages if they exist
PKG_RT="libcusparselt0-cuda-${CUDA_MAJOR}"
PKG_DEV="libcusparselt0-dev-cuda-${CUDA_MAJOR}"

if ! apt-cache show "${PKG_RT}" >/dev/null 2>&1; then
  echo "[cuSPARSELt] Fallback to generic package names"
  PKG_RT="libcusparselt0"
  PKG_DEV="libcusparselt0-dev"
fi

echo "[cuSPARSELt] Installing: ${PKG_RT} ${PKG_DEV}"
sudo apt-get install -y "${PKG_RT}" "${PKG_DEV}"

echo "[cuSPARSELt] Installed packages:"
dpkg -l | grep -Ei 'cusparselt|libcusparselt' || true

echo "[cuSPARSELt] Refreshing linker cache..."
sudo ldconfig

echo "[cuSPARSELt] Done."