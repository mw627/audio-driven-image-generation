import os

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_NAME = "qwen3-omni-30b-a3b-captioner"

SUPPORTED_AUDIO_EXT = [".mp3", ".wav", ".aac", ".flac", ".ogg", ".amr", ".3gp", ".3gpp", ".m4a"]

MAX_AUDIO_MB = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

# Prompt Adapter 使用的 LLM
PROMPT_ADAPTER_MODEL = "qwen3-next-80b-a3b-instruct"

# Stable Diffusion 通用高质量后缀
PROMPT_SUFFIX = (
    "masterpiece, best quality, ultra-detailed, clear, 8k, realist style"
)

# API Key & Base URL
PROMPT_ADAPTER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# source venv/bin/activate
# source /etc/network_turbo