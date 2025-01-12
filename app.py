import gradio as gr
from markitdown import MarkItDown
from PIL import Image
from google import genai
from google.genai import types

DEFAULT_SYSTEM_PROMPT = """당신은 과학/공학 분야의 전문가이자 소셜 미디어 마케팅 전문가입니다.
제공된 배경 지식과 이미지를 바탕으로 인스타그램에 적합한 홍보글을 작성해야 합니다.

다음 지침을 따라주세요:
1. 전문적이면서도 대중이 이해하기 쉬운 언어를 사용하세요
2. 핵심 내용을 먼저 전달하고, 세부 내용을 부연 설명하세요
3. 적절한 이모지를 활용하여 가독성을 높이세요
4. 관련 해시태그를 5-10개 포함하세요
5. 전체 글자 수는 500자 이내로 작성하세요
6. Plain text로 작성하고, Markdown 형식을 사용하지 마세요"""


PROMPT_TEMPLATE = """배경 지식:
{background_knowledge}

이미지 설명:
{image_description}

위 내용을 바탕으로 인스타그램 홍보 포스트를 작성해주세요."""


def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    result = MarkItDown().convert(pdf_file)
    return f"<article>{result.text_content}</article>"


def create_instagram_post(
    model_choice, api_token, system_prompt, pdf_files, image_nparray, image_description
):
    background_articles = []
    for pdf_file in pdf_files:
        background_articles.append(extract_text_from_pdf(pdf_file))

    background_knowledge = "\n".join(background_articles)

    image = Image.fromarray(image_nparray)
    image.thumbnail([512, 512])

    prompt = PROMPT_TEMPLATE.format(
        background_knowledge=background_knowledge, image_description=image_description
    )
    client = genai.Client(api_key=api_token)
    response = client.models.generate_content(
        model=model_choice,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            candidate_count=1,
        ),
    )
    return response.text


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
with gr.Blocks() as iface:
    gr.Markdown("# 과학/공학 인스타그램 포스트 생성기")

    with gr.Row():
        # 첫 번째 컬럼: 모델 설정
        with gr.Column():
            gr.Markdown("### 모델 설정")
            model_choice = gr.Dropdown(
                choices=["gemini-2.0-flash-exp"],
                label="LLM 모델 선택",
                value="gemini-2.0-flash-exp",
            )
            api_token = gr.Textbox(
                label="API 토큰",
                placeholder="선택한 모델의 API 토큰을 입력하세요",
                type="password",
            )
            system_prompt = gr.Textbox(
                label="System Prompt",
                placeholder="LLM 모델에게 전달할 시스템 프롬프트를 입력하세요",
                lines=5,
                value=DEFAULT_SYSTEM_PROMPT,
            )
            pdf_files = gr.File(
                file_count="multiple", label="PDF 파일들 (논문, 발표자료 등)"
            )

        # 두 번째 컬럼: 이미지 및 설명
        with gr.Column():
            gr.Markdown("### 이미지 및 설명")
            image = gr.Image(label="인스타그램 업로드 이미지")
            image_description = gr.Textbox(
                label="이미지 설명",
                placeholder="이미지에 대한 설명을 입력하세요",
                lines=3,
            )
            submit_btn = gr.Button("포스트 생성", variant="primary")

        # 세 번째 컬럼: 출력
        with gr.Column():
            gr.Markdown("### 생성된 포스트")
            output = gr.Textbox(
                label="생성된 인스타그램 포스트", lines=20, show_copy_button=True
            )

    submit_btn.click(
        fn=create_instagram_post,
        inputs=[
            model_choice,
            api_token,
            system_prompt,
            pdf_files,
            image,
            image_description,
        ],
        outputs=output,
    )


if __name__ == "__main__":
    iface.launch()
