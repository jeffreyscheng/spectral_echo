# pick an env name
ENV=spectral-echo

# create a fresh env (override channels so you don't inherit a broken channel config)
conda create -y -n "$ENV" --override-channels -c conda-forge -c defaults python=3.12 pip

# install torch from the CUDA 13.0 wheel index
conda run -n "$ENV" python -m pip install -U pip
conda run -n "$ENV" python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch

# install the rest of your requirements (do NOT re-install torch here)
conda run -n "$ENV" python -m pip install numpy tqdm scipy huggingface-hub

# sanity check
conda run -n "$ENV" python - <<'PY'
import torch, subprocess
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print(subprocess.check_output(["nvcc","--version"]).decode().splitlines()[-1])
PY
