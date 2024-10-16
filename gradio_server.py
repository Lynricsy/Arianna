import os
import time

def manage_response_file():
    if os.path.exists("response.txt"):
        current_time = int(time.time())
        new_name = f"response_{current_time}.txt"
        os.rename("response.txt", new_name)
    # 创建一个新的空文件
    open("response.txt", 'w').close()

# 在其他模块导入之前调用该函数
manage_response_file()
import gradio as gr
import random
from recurrentgpt import RecurrentGPT
from human_simulator import Human
from sentence_transformers import SentenceTransformer
from utils import get_init, parse_instructions
import re

# from urllib.parse import quote_plus
# from pymongo import MongoClient

# uri = "mongodb://%s:%s@%s" % (quote_plus("xxx"),
#                               quote_plus("xxx"), "localhost")
# client = MongoClient(uri, maxPoolSize=None)
# db = client.recurrentGPT_db
# log = db.log

_CACHE = {}


# Build the semantic search model
embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

def init_prompt(novel_type, description):
    if description == "":
        description = ""
    else:
        description = " about " + description
    return f"""
Please write a {novel_type} novel{description} with 50 chapters. Follow the format below precisely:

Begin with the name of the novel.
Next, write an outline for the first chapter. The outline should describe the background and the beginning of the novel.
Write the first three paragraphs with their indication of the novel based on your outline. Write in a novelistic style and take your time to set the scene.
Write a summary that captures the key information of the three paragraphs.
Finally, write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible, interesting continuation of the story.
The output format should follow these guidelines:
名称:  <name of the novel>
概述:  <outline for the first chapter>
段落 1: <content for paragraph 1>
段落 2: <content for paragraph 2>
段落 3: <content for paragraph 3>
总结: <content of summary>
选项支 1: <content for instruction 1>
选项支 2: <content for instruction 2>
选项支 3: <content for instruction 3>

Make sure to be precise and follow the output format strictly.
非常重要！请在输出信息中除了规定的格式之外全部使用中文（除了人名），注意要符合中文母语的语法和用词习惯。
语言尽量生动活泼，不要过于生僻或者过于正式。Galgame的剧情都非常非常长，一般需要20~30个小时才能读完，动辄几十上百万字。
你的总体剧情应该发展比较缓慢，充满悬念和节外生枝，剧情发展缓慢的同时，不能平淡无聊，增加较多的小插曲是很好的选择，也可以用来更好的刻画人物性格。
也就是说，你以一个固定的角色（主角）（一般是男性）视角写故事，你的故事情节中只能出现对话（占绝大部分篇幅）、环境描写（在每一行环境描写使用<environment>标签包裹）和主角的心理活动（在每一行主角的心理活动使用<hearty>标签包裹）。
注意一定要大量的角色语言。
角色的语言要富有个性，生动灵活，不要过于死板。
每行角色语言这样写:”<say><character>某人</character>xxxx</say>“也就是说，每句语言需要用<say>标签包裹，而说这句话的角色名字要用<character>标签包裹。
每一句角色语言、环境描写、心理描写都要单独一行，单独加上标签，类似一个XML剧本。
每个角色都要有自己的名字，不要用“ABC”“甲乙丙”"同学A同学B"这种符号化代称，也不要使用“班主任”“xx领导”等等代称，一切角色在该角色语言的冒号前面用完整的原名来指代，想表达这个角色的身份的话可以在他第一次出场或者其他时候用一两句话自我介绍或其他方式介绍。


"""

def init(novel_type, description, request: gr.Request):
    if novel_type == "":
        novel_type = "浪漫的爱情故事"
    global _CACHE
    cookie = request.headers.get('cookie', None)
    if not cookie:
        # 处理没有 cookie 的情况
        # 例如，生成一个唯一 ID 或使用会话标识符
        cookie = str(1111)  # 示例生成唯一 ID
    else:
        cookie = cookie.split('; _gat_gtag')[0]
    # prepare first init
    init_paragraphs = get_init(text=init_prompt(novel_type,description))
    # print(init_paragraphs)
    start_input_to_human = {
        'output_paragraph': init_paragraphs['Paragraph 3'],
        'input_paragraph': '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']]),
        'output_memory': init_paragraphs['Summary'],
        "output_instruction": [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }

    _CACHE[cookie] = {"start_input_to_human": start_input_to_human,
                      "init_paragraphs": init_paragraphs}
    written_paras = f"""标题: {init_paragraphs['name']}

梗概: {init_paragraphs['Outline']}

段落:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']])
    # short memory, long memory, current written paragraphs, 3 next instructions
    return start_input_to_human['output_memory'], long_memory, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def step(short_memory, long_memory, instruction1, instruction2, instruction3, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    # print(request.headers.get('cookie'))
    cookie = request.headers.get('cookie', None)
    if not cookie:
        # 处理没有 cookie 的情况
        # 例如，生成一个唯一 ID 或使用会话标识符
        cookie = str(1111)  # 示例生成唯一 ID
    else:
        cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = [
            instruction1, instruction2, instruction3]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        #randomly select one instruction out of three
        instruction_index = random.randint(0,2)
        output['output_instruction'] = [instruction1, instruction2, instruction3][instruction_index]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = [[v] for v in writer.long_memory]
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], human.output['output_instruction'], *writer.output['output_instruction']


def controled_step(short_memory, long_memory, selected_instruction, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    # print(request.headers.get('cookie'))
    cookie = request.headers.get('cookie', None)
    if not cookie:
        # 处理没有 cookie 的情况
        # 例如，生成一个唯一 ID 或使用会话标识符
        cookie = str(1111)  # 示例生成唯一 ID
    else:
        cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]
    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = selected_instruction
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        output['output_instruction'] = selected_instruction
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], parse_instructions(writer.long_memory), current_paras + '\n\n' + writer.output['input_paragraph']+ '\n\n' + writer.output['output_paragraph'], *writer.output['output_instruction']


# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("选项支 ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan-1]
    return selected_plan


with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    gr.Markdown(
        """
    # Arianna

    ## Not only RecurrentGPT For AI-Assisted Galgame Writing

    ### 輝け、Galgameの未来——
    ### いざAI Galgameへ向けて――我らペガサス組、しゅっぱーつ！

    Co-pilot for writing a Galgame story. The AI will generate a story based on the input and provide three options for the next step. The user can choose one of the options to continue the story. The AI will then generate the next step based on the user's choice.

    Developed by **Shirakami Lynricsy**
    """)
    with gr.Tab("自动剧情（呜呜暂时不太好用）"):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="请输入文本", placeholder="可以自己填写或者从EXamples中选择一个填入")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="剧情简介（非必选项）")
                btn_init = gr.Button(
                    "开始故事！", variant="primary")
                gr.Examples(["科幻故事", "青春伤痛文学", "爱到死去活来", "搞笑",
                            "幽默", "鬼故事", "喜剧", "童话", "魔法世界", ], inputs=[novel_type])
                written_paras = gr.Textbox(
                    label="文章内容", max_lines=21, lines=21)
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### 剧情模型\n")
                    short_memory = gr.Textbox(
                        label="短期记忆 (可编辑)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(
                        label="长期记忆 (可编辑)", max_lines=6, lines=6)
                    # long_memory = gr.Dataframe(
                    #     # label="Long-Term Memory (editable)",
                    #     headers=["Long-Term Memory (editable)"],
                    #     datatype=["str"],
                    #     row_count=3,
                    #     max_rows=3,
                    #     col_count=(1, "fixed"),
                    #     type="array",
                    # )
                with gr.Group():
                    gr.Markdown("### 选项模型\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="选项支1(可编辑)", max_lines=4, lines=4)
                        instruction2 = gr.Textbox(
                            label="选项支2(可编辑)", max_lines=4, lines=4)
                        instruction3 = gr.Textbox(
                            label="选项支3(可编辑)", max_lines=4, lines=4)
                    selected_plan = gr.Textbox(
                        label="选项说明 (来自上一步)", max_lines=2, lines=2)

                btn_step = gr.Button("下一步", variant="primary")

        btn_init.click(init, inputs=[novel_type, description], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(step, inputs=[short_memory, long_memory, instruction1, instruction2, instruction3, written_paras], outputs=[
            short_memory, long_memory, written_paras, selected_plan, instruction1, instruction2, instruction3])

    with gr.Tab("自选择剧情（推荐！剧情由你控制！）"):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="请输入文本", placeholder="可以自己填写或者从EXamples中选择一个填入")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="剧情简介（非必选项）")
                btn_init = gr.Button(
                    "开始故事！", variant="primary")
                gr.Examples(["科幻小说", "爱情小说", "推理小说", "奇幻小说",
                            "玄幻小说", "恐怖", "悬疑", "惊悚", "武侠小说", ], inputs=[novel_type])
                written_paras = gr.Textbox(
                    label="文章内容在这里~ (可编辑)", max_lines=23, lines=23)
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### 剧情记忆~\n")
                    short_memory = gr.Textbox(
                        label="这是短期记忆~ (可编辑)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(
                        label="这是长期记忆记忆~ (可编辑)", max_lines=6, lines=6)
                with gr.Group():
                    gr.Markdown("### 下一步会如何呢喵~\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="选项支1", max_lines=3, lines=3, interactive=False)
                        instruction2 = gr.Textbox(
                            label="选项支2", max_lines=3, lines=3, interactive=False)
                        instruction3 = gr.Textbox(
                            label="选项支3", max_lines=3, lines=3, interactive=False)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(["选项支 1", "选项支 2", "选项支 3"], label="选项支 选择",)
                                                    #  info="Select the instruction you want to revise and use for the next step generation.")
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="这里是你选择的剧情呢~随你的想法修改一下吧？说不定可以来个巨大的转折，把读者惊呆呢？", max_lines=5, lines=5)

                btn_step = gr.Button("下一步", variant="primary")

        btn_init.click(init, inputs=[novel_type, description], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(controled_step, inputs=[short_memory, long_memory, selected_instruction, written_paras], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        selected_plan.select(on_select, inputs=[
                             instruction1, instruction2, instruction3], outputs=[selected_instruction])

    demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_port=8005, share=True,
                server_name="0.0.0.0", show_api=False)