import os
import requests
import tempfile
from urllib.parse import urlparse
from pydub import AudioSegment


SUPPORTED_EXT = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".amr", ".3gp"}
MAX_FILE_MB = 10
MAX_DURATION_SEC = 40 * 60  # 40分钟


class AudioCheckError(Exception):
    """音频校验错误（用于前端提示）"""
    pass


def get_audio_info(audio_path: str):
    """读取音频并返回对象与时长"""
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio) / 1000
    return audio, duration


def validate_audio_format(audio_path: str):
    """音频格式检测"""
    ext = os.path.splitext(audio_path)[1].lower()

    if ext not in SUPPORTED_EXT:
        raise AudioCheckError(
            f"不支持的音频格式：{ext}。支持格式：WAV / MP3 / AAC / FLAC / OGG / AMR / 3GP"
        )


def validate_audio_size(audio_path: str):
    """音频大小检测"""
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        raise AudioCheckError(
            f"音频文件大小为 {size_mb:.2f}MB，超过 10MB 限制，请压缩音频后重新上传"
        )


def trim_audio_if_needed(audio_path: str, strategy="middle"):
    """
    若音频超过40分钟则自动裁剪

    strategy:
        head   -> 前段
        middle -> 中段（默认）
        tail   -> 后段
    """

    audio, duration = get_audio_info(audio_path)

    if duration <= MAX_DURATION_SEC:
        return audio_path

    print(f"[AudioCheck] 音频时长 {duration:.1f}s 超过限制，进行自动裁剪")

    if strategy == "head":
        clipped = audio[:MAX_DURATION_SEC * 1000]

    elif strategy == "tail":
        clipped = audio[-MAX_DURATION_SEC * 1000:]

    else:
        mid = len(audio) // 2
        half = MAX_DURATION_SEC * 500
        clipped = audio[mid - half: mid + half]

    new_path = audio_path.replace(".", "_trimmed.")

    clipped.export(new_path, format="mp3")

    return new_path


def download_audio_from_url(url: str) -> str:
    """下载网络音频到临时文件"""
    try:
        # === 第一步：HEAD 请求预检，不下载内容 ===
        try:
            head = requests.head(url, timeout=10, allow_redirects=True)
            content_length = head.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > MAX_FILE_MB:
                    raise AudioCheckError(
                        f"文件大小超过 10MB 限制（当前 {size_mb:.2f}MB），请压缩后重新上传"
                    )
        except AudioCheckError:
            raise
        except Exception:
            # HEAD 请求失败不阻断流程，继续尝试下载
            pass

        # === 第二步：流式下载，边下载边累计大小 ===
        response = requests.get(url, stream=True, timeout=15)

        if response.status_code != 200:
            raise AudioCheckError(
                f"无法下载音频，请检查URL是否有效（HTTP {response.status_code}）"
            )

        # 再次从响应头检查（GET 响应头可能和 HEAD 不同）
        content_length = response.headers.get("Content-Length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                raise AudioCheckError(
                    f"文件大小超过 10MB 限制（当前 {size_mb:.2f}MB），请压缩后重新上传"
                )

        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext not in SUPPORTED_EXT:
            raise AudioCheckError(
                f"不支持的音频格式：{ext}。支持格式：WAV / MP3 / AAC / FLAC / OGG / AMR / 3GP"
            )

        # === 第三步：写入临时文件，同时累计已下载大小 ===
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        downloaded_mb = 0

        for chunk in response.iter_content(chunk_size=1024 * 256):  # 256KB 每块
            temp_file.write(chunk)
            downloaded_mb += len(chunk) / (1024 * 1024)

            # 超限立即中断，不等下载完
            if downloaded_mb > MAX_FILE_MB:
                temp_file.close()
                os.unlink(temp_file.name)  # 删除不完整的临时文件
                raise AudioCheckError(
                    f"文件大小超过 10MB 限制，请压缩后重新上传"
                )

        temp_file.close()
        print(f"[AudioCheck] 下载完成，临时文件：{temp_file.name}")
        return temp_file.name

    except AudioCheckError:
        raise
    except requests.exceptions.RequestException as e:
        raise AudioCheckError(f"音频URL请求失败，请检查网络连接（{str(e)}）")


def load_audio_input(audio_input: str):
    """
    统一加载输入

    支持：
        1 本地文件路径
        2 网络URL
    """

    if audio_input.startswith("http://") or audio_input.startswith("https://"):

        print("[AudioCheck] 检测到URL音频，开始下载")

        audio_path = download_audio_from_url(audio_input)

    else:

        audio_path = audio_input

    if not os.path.exists(audio_path):
        raise AudioCheckError("音频文件不存在或无法访问")

    return audio_path


def process_audio_input(audio_input: str) -> str:
    """
    Audio Captioning 模块统一入口

    输入：
        本地路径 或 URL

    输出：
        可直接送入 Captioning 模型的音频路径
    """

    audio_path = load_audio_input(audio_input)

    validate_audio_format(audio_path)

    validate_audio_size(audio_path)

    audio_path = trim_audio_if_needed(audio_path)

    return audio_path