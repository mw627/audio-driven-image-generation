from openai import OpenAI
from config import settings


STAGE1_SYSTEM_PROMPT = """
You are an audio scene understanding assistant.

Your task is to convert an audio description into a structured scene summary.

Output format (strictly follow this):
SUBJECT: [the single most prominent sound source or actor]
ACTION: [what the subject is doing]
ENVIRONMENT: [where this is happening]
MOOD: [the emotional atmosphere in one word]

Rules:
- SUBJECT must be a concrete noun (person, animal, object, nature element).
- Choose only ONE subject — the most dominant one in the audio.
- Do NOT use audio/technical terms (e.g. "sound", "audio", "frequency", "recording").
- Do NOT add visual or artistic descriptions.
- Each field must be filled, no field can be empty.
- Use plain English only.
""".strip()


STAGE2_SYSTEM_PROMPT = """
You are a professional Stable Diffusion prompt engineer.

You will receive a structured scene summary with four fields: SUBJECT, ACTION, ENVIRONMENT, MOOD.

Your task is to convert it into a single Stable Diffusion image prompt.

Rules:
- The SUBJECT field is the mandatory visual focus — it MUST appear as the first element in your prompt.
- Follow this structure strictly: [SUBJECT] [ACTION], [ENVIRONMENT], [MOOD-driven lighting and color palette]
- Use vivid, concrete visual adjectives and nouns.
- Do NOT change or replace the subject.
- Do NOT add people, characters or objects not implied by the scene summary.
- Do NOT include any explanation or reasoning.
- Output ONE line only, in English, under 75 tokens.
""".strip()


class PromptAdapter:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.PROMPT_ADAPTER_BASE_URL,
        )
        self.model = settings.PROMPT_ADAPTER_MODEL

    def _call_llm(self, system_prompt: str, user_input: str, max_tokens=120, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def adapt(self, audio_caption: str) -> str:
        """
        Two-stage prompt adaptation:
        Audio Caption -> Scene Summary -> Visual Prompt
        """

        # Stage 1: 听懂
        scene_summary = self._call_llm(
            STAGE1_SYSTEM_PROMPT,
            audio_caption,
            max_tokens=120,
            temperature=0.3,  # Stage 1 更注重准确性，降低温度
        )

        # Stage 2: 画画
        visual_prompt = self._call_llm(
            STAGE2_SYSTEM_PROMPT,
            scene_summary,
            max_tokens=120,
            temperature=0.4,  # Stage 2 鼓励创造性，适当提高温度
        )

        final_prompt = f"{visual_prompt}"

        return final_prompt
