#!/bin/bash

# Script to install IBM Aspera Connect (ascp) on Linux using .tar.gz for all distros

# Exit on any error
set -e

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    echo "This script should not be run as root for .tar.gz installations."
    exit 1
fi

# Variables
ASPERA_URL="https://d3gcli72yxqn2z.cloudfront.net/downloads/connect/latest/bin/ibm-aspera-connect_4.2.16.884-HEAD_linux_x86_64.tar.gz"
DOWNLOAD_DIR="$HOME/Downloads"
INSTALL_DIR="$HOME/.aspera/connect"
ASPERA_VERSION="4.2.16.884-HEAD"
PACKAGE_NAME="ibm-aspera-connect"
TARBALL_NAME="${PACKAGE_NAME}_${ASPERA_VERSION}_linux_x86_64.tar.gz"

# Function to install dependencies
install_dependencies() {
    # Check for wget or curl
    if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
        echo "Error: wget or curl is required to download the package."
        exit 1
    fi

    # Install libssl-dev or equivalent if possible
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        case $DISTRO in
            ubuntu|debian)
                sudo apt-get update
                sudo apt-get install -y libssl-dev
                ;;
            centos|rhel|fedora)
                sudo yum install -y openssl-libs || sudo dnf install -y openssl-libs
                ;;
            *)
                echo "Unsupported distribution for automatic dependency installation."
                echo "Please manually install libssl or equivalent for your distro."
                ;;
        esac
    else
        echo "Cannot detect distribution. Skipping dependency installation."
        echo "Please ensure libssl or equivalent is installed."
    fi
}

# Function to download and install Aspera Connect
install_ascp() {
    mkdir -p "$DOWNLOAD_DIR"
    cd "$DOWNLOAD_DIR"

    echo "Downloading Aspera Connect from $ASPERA_URL..."
    if command -v wget >/dev/null 2>&1; then
        wget -O "$TARBALL_NAME" "$ASPERA_URL" || { echo "Download failed!"; exit 1; }
    elif command -v curl >/dev/null 2>&1; then
        curl -o "$TARBALL_NAME" "$ASPERA_URL" || { echo "Download failed!"; exit 1; }
    else
        echo "Error: wget or curl not found!"
        exit 1
    fi

    # Verify download
    if [ ! -f "$TARBALL_NAME" ]; then
        echo "Error: Failed to download $TARBALL_NAME!"
        exit 1
    fi

    # List contents of tarball for debugging
    echo "Listing contents of $TARBALL_NAME..."
    tar -tzf "$TARBALL_NAME"

    echo "Extracting $TARBALL_NAME..."
    tar -zxvf "$TARBALL_NAME" || { echo "Extraction failed!"; exit 1; }

    # Find the .sh file (more flexible wildcard)
    SH_FILE=$(find . -maxdepth 1 -type f -name "*.sh" | head -n 1)
    if [ -z "$SH_FILE" ]; then
        echo "Error: No .sh file found in extracted contents!"
        ls -la
        exit 1
    fi

    echo "Found installation script: $SH_FILE"
    chmod +x "$SH_FILE"
    echo "Installing Aspera Connect..."
    ./"$SH_FILE" || { echo "Installation script failed!"; exit 1; }
}

# Function to verify installation
verify_installation() {
    ASCP_PATH="$INSTALL_DIR/bin/ascp"
    if [ -f "$ASCP_PATH" ]; then
        echo "Verifying ascp installation..."
        $ASCP_PATH --version
    else
        echo "ascp not found in $ASCP_PATH. Checking system path..."
        if command -v ascp >/dev/null 2>&1; then
            ascp --version
        else
            echo "Installation failed or ascp not in PATH."
            exit 1
        fi
    fi
}

# Function to update PATH
update_path() {
    if [ -d "$INSTALL_DIR/bin" ] && ! echo "$PATH" | grep -q "$INSTALL_DIR/bin"; then
        echo "Adding $INSTALL_DIR/bin to PATH..."
        echo "export PATH=\$PATH:$INSTALL_DIR/bin" >> "$HOME/.bashrc"
        export PATH=$PATH:$INSTALL_DIR/bin
        source "$HOME/.bashrc"
    fi
}

# Main execution
echo "Starting IBM Aspera Connect installation..."
install_dependencies
install_ascp
verify_installation
update_path

echo "Installation completed successfully!"
echo "You can use ascp for file transfers, e.g.:"
echo "ascp -P 33001 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh user@host:/remote/path /local/path"
echo "You may need to restart your terminal or run 'source ~/.bashrc' to update your PATH."