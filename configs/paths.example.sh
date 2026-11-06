# Copy outside Git, replace paths with writable local directories.
export V7_DATA_ROOT="$HOME/.local/share/v7-reward"
export V7_RUNS_ROOT="$V7_DATA_ROOT/runs/v7"
export V7_REWARD_REPO=/your/workspace/v7-reward
export V7_REWARD_PYTHON=/your/src/sole-r1-model/reward_server/.venv/bin/python
export PYTHON="$V7_REWARD_PYTHON"  # v3 uses the same SOLE-R1 environment
export V7_POLICY_PYTHON=/your/environments/policy/bin/python
export V7_RLINF_TRAINING=/your/workspace/RLinf-training
export V7_COLLECTION_REPO=/your/workspace/RLinf-collection
export V7_BASE_MODEL="$V7_DATA_ROOT/models/Qwen3-VL-8B-Instruct"
export V7_REWARD_MODEL="$V7_DATA_ROOT/models/sole-r1-v7-full"
# API generation only. Never commit the secret file.
export V7_API_BASE_URL=https://your-provider.example/v1
export V7_API_KEY_FILE="$V7_DATA_ROOT/private/api_key"
export V7_POLICY_MODEL="$V7_DATA_ROOT/models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
export V7_ARTIFACTS="$V7_DATA_ROOT/runs/rlinf"
export V7_TOKENIZER_ASSET="$V7_DATA_ROOT/assets/paligemma_tokenizer.model"
export V7_RAY_ROOT="$V7_DATA_ROOT/ray"

# Keep LIBERO configurations separate when several environments coexist.
export LIBERO_CONFIG_PATH="$V7_DATA_ROOT/config/libero-policy"
