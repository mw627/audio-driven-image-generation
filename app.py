import os
import gradio as gr

from services.captioner import AudioCaptioner
from services.prompt_adapter_double import PromptAdapter
from services.image_generator import ImageGenerator
from utils.audio_check1 import process_audio_input, AudioCheckError

STYLE_OPTIONS = {
    "写实风格": "photorealistic, 8k, ultra detailed",
    "油画风格": "oil painting, impressionist, textured canvas",
    "水彩风格": "watercolor painting, soft edges, pastel colors",
    "动漫风格": "anime style, Studio Ghibli, vibrant colors",
    "素描风格": "pencil sketch, black and white, hand drawn",
}

# =========================
# 初始化模块
# =========================

captioner = AudioCaptioner()
prompt_adapter = PromptAdapter()
image_generator = ImageGenerator()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 核心推理函数
# =========================

def audio_to_image(
    audio_file,
    audio_url,
    style,
    steps,
    guidance,
    width,
    height,
    seed
):
    if audio_file is not None:
        audio_input = audio_file
    elif audio_url and audio_url.strip():
        audio_input = audio_url.strip()
    else:
        return "", []

    try:
        audio_path = process_audio_input(audio_input)
        caption = captioner.caption(audio_path)
        prompt = prompt_adapter.adapt(caption)

        style_suffix = STYLE_OPTIONS.get(style, "")
        if style_suffix:
            prompt = f"{prompt}, {style_suffix}"

        seed = int(seed)
        seed = None if seed == -1 else seed

        image_paths = image_generator.generate_batch(
            prompt=prompt,
            output_dir=OUTPUT_DIR,
            num_images=3,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            seed=seed,
        )

        return prompt, image_paths

    except Exception as e:
        return f"Error: {str(e)}", []


# =========================
# Gradio UI
# =========================

with gr.Blocks(title="Audio2Image System") as demo:

    gr.Markdown(
        """
        # 音频驱动图像生成系统
        上传音频，系统自动识别场景并生成对应图像。
        """
    )

    gr.Markdown(
        """
        ## 使用说明：
        1. 点击上传按钮选择音频文件（支持mp3, wav, flac等）。
        2. 或在下方输入音频URL。
        3. 当用户同时上传音频文件和出入音频URL时，图片生成将以音频文件为准。
        4. 系统会自动识别音频内容并生成对应图像。
        5. 生成的Prompt用于Stable Diffusion图像生成任务。
        """
    )

    with gr.Row():


        with gr.Column(scale=1):

            audio_input = gr.Audio(
                label="上传音频",
                type="filepath"
            )
            
            audio_url_box = gr.Textbox(
                label="音频 URL",
                placeholder="输入音频文件URL"
            )

            style_dropdown = gr.Dropdown(
                choices=list(STYLE_OPTIONS.keys()),
                value="写实风格",
                label="图像风格"
            )

            generate_btn = gr.Button("生成图像")

            gr.Markdown("### Stable Diffusion参数")

            steps_slider = gr.Slider(
                10,
                50,
                value=30,
                step=1,
                label="推理步数"
            )

            guidance_slider = gr.Slider(
                1,
                15,
                value=7.5,
                step=0.5,
                label="Guidance Scale"
            )

            width_slider = gr.Slider(
                256,
                768,
                value=512,
                step=64,
                label="图像宽度"
            )

            height_slider = gr.Slider(
                256,
                768,
                value=512,
                step=64,
                label="图像高度"
            )

            seed_box = gr.Textbox(
                value="-1",
                label="随机种子（-1表示随机）"
                )



        with gr.Column(scale=1):


            prompt_output = gr.Textbox(
                label="生成图像Prompt",
                lines=4
            )

            image_output = gr.Gallery(
                label="生成图像",
                columns=3,        # 三张并排显示
                height="auto",
            )

    # =========================
    # 绑定按钮
    # =========================

    generate_btn.click(
        fn=audio_to_image,
        inputs=[
            audio_input,
            audio_url_box,
            style_dropdown,
            steps_slider,
            guidance_slider,
            width_slider,
            height_slider,
            seed_box
        ],
        outputs=[
            prompt_output,
            image_output
        ]
    )


# =========================
# 启动服务
# =========================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False
    )