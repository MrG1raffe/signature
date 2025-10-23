# Signatures

Implementation of tensor algebra methods for path signatures and exponentially fading-memory signature and its applications.

---

## Installation
Clone the repository
```bash
git clone https://github.com/MrG1raffe/signature
cd signature
```

(Optionally) create a new environment
```bash
conda env create -f environment.yml
conda activate signature
```

Install the package
```bash
pip isntall .
```

### Notes on iisignature
For the iisignature module to work correctly, make sure you have C++ compilers (g++, gcc) installed on your system.

On Ubuntu or Debian-based systems:
```bash
sudo apt install build-essential
```
On macOS (with Xcode Command Line Tools):
```bash
xcode-select --install
```