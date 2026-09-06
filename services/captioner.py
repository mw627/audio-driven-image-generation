import os
import dashscope
from config.settings import MODEL_NAME


class AudioCaptioner:
    """
    Audio Captioning 服务
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置")

        self.model = MODEL_NAME

    def _build_messages(self, audio_path: str):

        return [
            {
                "role": "user",
                "content": [
                    {"audio": audio_path}
                ]
            }
        ]

    def caption(self, audio_path: str) -> str:
        """
        输入：已经通过 AudioCheck 的音频路径
        """

        if not os.path.exists(audio_path):
            raise RuntimeError("音频文件不存在")

        messages = self._build_messages(audio_path)

        try:

            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model,
                messages=messages
            )

            return self._parse_response(response)

        except Exception as e:
            raise RuntimeError(f"音频描述生成失败：{str(e)}")

    @staticmethod
    def _parse_response(response):

        try:
            return response["output"]["choices"][0]["message"].content[0]["text"]
        except Exception:
            raise RuntimeError("模型返回格式异常")