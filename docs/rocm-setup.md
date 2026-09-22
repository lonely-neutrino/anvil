# Anvil AMD/ROCm setup notes

This records the GPU setup used for Anvil on the AMD machine. The important
point is that the repository's normal `pyproject.toml`/`uv.lock` dependency
resolution is not the source of the working PyTorch installation.

The repository declares only `torch>=2.12.1`, and the lockfile resolves a
standard PyTorch build with CUDA/NVIDIA packages. The working environment is a
separate ROCm build installed into `.venv`. Since `.venv/` is gitignored, this
distinction is easy to lose when recreating the checkout.

## Known-good environment snapshot

The current `.venv` contains:

```text
Python:                  3.12.3
uv:                      0.12.7

torch:                   2.13.0+rocm7.14.0
amd-torch-device-gfx1101 2.13.0+rocm7.14.0
triton:                  3.8.0+git4cff872c.rocm7.14.0

rocm:                    7.14.0
rocm-sdk-core:           7.14.0
rocm-sdk-device-gfx1101: 7.14.0
rocm-sdk-libraries:      7.14.0
```

This setup was for and worked with an **AMD Radeon RX 7800 XT (16 GB)**.

The GPU target is `gfx1101`. PyTorch identifies this as a ROCm build:

```text
torch.version.hip = 7.14.60850
torch.version.cuda = None
USE_CUDA = OFF
USE_ROCM = ON
```

The package names `cuda` and `torch.cuda` are still used by PyTorch's ROCm
backend. They are not evidence that the NVIDIA build should be installed.

## Host prerequisites

The shell history strongly suggests this machine is using WSL2 with the WSL
GPU bridge. Before troubleshooting Python, check the host integration:

```bash
cat /etc/os-release | grep -E 'PRETTY_NAME|VERSION_ID'
ls -l /dev/dxg
ls -l /usr/lib/wsl/lib/libdxcore.so
command -v rocminfo
rocminfo | grep -A15 -B5 gfx1101
command -v hipinfo || find /opt/rocm* -type f -name hipinfo -print 2>/dev/null
find /opt -name 'librocdxg.so*' -print 2>/dev/null
find /opt/rocm -name 'libhsa-runtime64.so*' -ls 2>/dev/null
```

The shell history also records installing `libnuma1`, which was useful for
ROCm/rocSHMEM startup:

```bash
sudo apt update
sudo apt install -y libnuma1
```

`rocminfo` should be able to see the GPU and report `gfx1101`. If it cannot,
fix the WSL/ROCm host integration before reinstalling Python packages.

## Recreate the Python environment

Install `uv` if necessary:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

From the Anvil checkout, create the environment while explicitly excluding
the lockfile's PyTorch package:

```bash
cd ~/mtg-ai/anvil
uv sync --locked --no-install-package torch
```

Then install the AMD PyTorch build from AMD's ROCm wheel index:

```bash
uv pip install \
  --python .venv/bin/python \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch[device-gfx1101]==2.13.0+rocm7.14.0"
```

The history shows a follow-up install of the device package explicitly. If it
is missing after the PyTorch install, run:

```bash
uv pip install \
  --python .venv/bin/python \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "amd-torch-device-gfx1101==2.13.0+rocm7.14.0"
```

The exact AMD index URL and package specifications above were recovered from
the shell history. The original package-resolution metadata was not committed
to the repository, so retain these pins rather than relying on an unqualified
`torch` installation.

## Verify the installation

Use the project interpreter directly:

```bash
source .venv/bin/activate
which python
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA runtime:", torch.version.cuda)
print("Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    print(f"Testing {dtype}...")
    x = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    y = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    z = x @ y
    torch.cuda.synchronize()
    print("  shape:", z.shape)
    print("  device:", z.device)
    print("  finite:", torch.isfinite(z).all().item())
PY
```

Then check dependencies and the repository:

```bash
uv pip check --python .venv/bin/python
uv pip list --python .venv/bin/python | grep -E 'torch|gfx1101|rocm|triton'
python -m pytest -q
```

For a future rebuild, save the complete installed package set before changing
anything:

```bash
uv pip freeze --python .venv/bin/python > /tmp/anvil-rocm-freeze.txt
```

## How to launch Anvil

Use either the activated environment or the interpreter path explicitly:

```bash
source .venv/bin/activate
python -m anvil.bridge.server --mode model --device cuda:0 ...
```

The repository's experiment scripts generally use:

```bash
./.venv/bin/python
```

Use `cuda`/`cuda:0` for PyTorch device arguments. ROCm PyTorch exposes the
AMD accelerator through that API, and changing it to `hip` or `rocm` would not
match the code's expected device interface.

Do not use plain `uv run` or an unrestricted `uv sync` for this checkout. They
can reconcile `.venv` with `uv.lock` and replace the ROCm build with the
lockfile's standard PyTorch build. The safe project-sync command is the one
above with `--no-install-package torch`, followed by the explicit AMD install.

## GPU memory and runtime details

The following are operational settings rather than package installation, but
they matter when reproducing the runs:

- RL accepts `--rl-seg`; the recorded runs commonly used `--rl-seg 64` and
  smoke tests used 32. Smaller values reduce activation peaks.
- An older GPU-cotenancy workaround set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator
  fragmentation. The current Anvil code only sets this automatically for
  non-ROCm PyTorch. Do not blindly export it for HIP.
- Checkpoints should be loaded on CPU first and then moved to `cuda:0`; this is
  the pattern used to avoid ROCm checkpoint-restore failures.
- The current scripts pass `--device cuda:0` to model serving and RL.
- The embedding CLI also uses the CUDA-compatible device name internally.

## Forge-side paths

Anvil also expects a Forge checkout. The scripts default to a sibling checkout:

```text
~/mtg-ai/forge
```

If it is elsewhere, set:

```bash
export FORGE_DIR=/absolute/path/to/forge
```

Constructed decks are normally expected under:

```text
~/.forge/decks/constructed
```

## Troubleshooting checklist

### `torch.version.hip` is `None`

The wrong PyTorch wheel is installed. Check:

```bash
uv pip list --python .venv/bin/python | grep -E 'torch|gfx1101|rocm|triton'
```

Reinstall the pinned AMD packages from the AMD index.

### `torch.version.hip` is present but `torch.cuda.is_available()` is false

Check `rocminfo`, `/dev/dxg`, the WSL GPU bridge, and `libnuma1`. Also confirm
that the `amd-torch-device-gfx1101` package is installed.

### `uv run` changes or breaks GPU support

Stop using it for this environment. Re-run the explicit ROCm installation
against `.venv/bin/python`; do not let the normal lockfile install PyTorch.

### ROCm reports missing `libnuma`

Install the system package:

```bash
sudo apt update
sudo apt install -y libnuma1
```

## What was recovered from command history

The Bash history contained the exact important setup steps:

1. Investigated WSL GPU files, ROCm libraries, `rocminfo`, and `gfx1101`.
2. Tested `torch[device-gfx1101]==2.12.0+rocm7.14.0` in a temporary
   `rocm-test` environment.
3. Installed `libnuma1`.
4. Moved to `torch[device-gfx1101]==2.13.0+rocm7.14.0`.
5. Installed `uv`.
6. Ran `uv sync --locked --no-install-package torch` in Anvil.
7. Installed the pinned AMD PyTorch build into `.venv` from
   `https://repo.amd.com/rocm/whl-multi-arch/`.
8. Explicitly installed `amd-torch-device-gfx1101==2.13.0+rocm7.14.0`.
9. Verified FP32, FP16, and BF16 matrix multiplication, ran `uv pip check`,
   and ran the test suite.

The history did not preserve the exact host GPU model, the host driver
installation procedure, or why the explicit device-package reinstall was
needed. The final installed package set and commands above are the strongest
available record.
