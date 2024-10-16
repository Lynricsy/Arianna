import re
import openai
from openai import OpenAI
from decouple import config

my_variable = config('MY_VARIABLE', default='default_value')

api_key = config('OPENAI_API_KEY', default='sk-xxxx')
api_base = config('OPENAI_BASE_URL', default='https://api.openai.com/v1')
def get_api_response(content: str, max_tokens=None):
    client = OpenAI(api_key=api_key,base_url=api_base)
    response = client.chat.completions.create(
        model='claude-3-5-sonnet-20240620',
        messages=[{
            'role': 'system',
            'content': '你是一个有帮助且富有创造力的Galgame(视觉小说)写作助手。Galgame的剧情都非常非常长，一般需要20~30个小时才能读完，动辄几十上百万字。你的总体剧情应该发展比较缓慢，充满悬念和节外生枝，剧情发展缓慢的同时，不能平淡无聊，增加较多的小插曲是很好的选择，也可以用来更好的刻画人物性格。也就是说，你以一个固定的角色（主角）（一般是男性）视角写故事，你的故事情节中只能出现对话（占绝大部分篇幅）、环境描写（在每一行环境描写前加入<environment>标签）和主角的心理活动（在每一行主角的心理活动前加入<hearty>标签）。注意一定要大量的角色语言。角色的语言要富有个性，生动灵活，不要过于死板。每行角色语言这样写:”<say><character>某人</character>:xxxx“也就是说，每句语言前面需要加上一个<say>标签，而说这句话的角色名字要用<character>标签包裹。每一句角色语言、环境描写、心理描写都要单独一行，单独加上标签，类似一个XML剧本。每个角色都要有自己的名字，不要用“ABC”“甲乙丙”"同学A同学B"这种符号化代称，也不要使用“班主任”“xx领导”等等代称，一切角色在该角色语言的冒号前面用完整的原名来指代，想表达这个角色的身份的话可以在他第一次出场或者其他时候用一两句话自我介绍或其他方式介绍。除了规定的格式内容外，请使用汉语来写小说。一切冒号使用半角（英文）冒号。无论多么小，多么不重要的人物，都一定要有自己的名字'
        }, {
            'role': 'user',
            'content': content,
        }],
        temperature=0.5,
        stream=True
    )
    final = ''
    for chunk in response:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                final += chunk.choices[0].delta.content
                # print(final)
    return final 

def get_content_between_a_b(a, b, text):
    match = re.search(f"{a}(.*?)\n{b}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # 处理找不到匹配内容的情况
        return "呜呜，返回格式有点问题呢~"



def get_init(init_text=None,text=None,response_file="response.txt"):
    """
    init_text: if the title, outline, and the first 3 paragraphs are given in a .txt file, directly read
    text: if no .txt file is given, use init prompt to generate
    """
    if not init_text:
        response = get_api_response(text)
        print(response)

        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Init output here:\n{response}\n\n")
    else:
        with open(init_text,'r',encoding='utf-8') as f:
            response = f.read()
        f.close()
    paragraphs = {
        "name":"",
        "Outline":"",
        "Paragraph 1":"",
        "Paragraph 2":"",
        "Paragraph 3":"",
        "Summary": "",
        "Instruction 1":"",
        "Instruction 2":"", 
        "Instruction 3":""    
    }
    paragraphs['name'] = get_content_between_a_b('名称:','概述:',response)
    
    paragraphs['Paragraph 1'] = get_content_between_a_b('段落 1:','段落 2:',response)
    paragraphs['Paragraph 2'] = get_content_between_a_b('段落 2:','段落 3:',response)
    paragraphs['Paragraph 3'] = get_content_between_a_b('段落 3:','总结',response)
    paragraphs['Summary'] = get_content_between_a_b('总结:','选项支 1',response)
    paragraphs['Instruction 1'] = get_content_between_a_b('选项支 1:','选项支 2:',response)
    paragraphs['Instruction 2'] = get_content_between_a_b('选项支 2:','选项支 3:',response)
    lines = response.splitlines()
    # content of Instruction 3 may be in the same line with I3 or in the next line
    if lines[-1] != '\n' and lines[-1].startswith('Instruction 3'):
        paragraphs['Instruction 3'] = lines[-1][len("Instruction 3:"):]
    elif lines[-1] != '\n':
        paragraphs['Instruction 3'] = lines[-1]
    # Sometimes it gives Chapter outline, sometimes it doesn't
    for line in lines:
        if line.startswith('Chapter'):
            paragraphs['Outline'] = get_content_between_a_b('概述:','Chapter',response)
            break
    if paragraphs['Outline'] == '':
        paragraphs['Outline'] = get_content_between_a_b('概述:','段落',response)


    return paragraphs

def get_chatgpt_response(model,prompt):
    response = ""
    for data in model.ask(prompt):
        response = data["message"]
    model.delete_conversation(model.conversation_id)
    model.reset_chat()
    return response


def parse_instructions(instructions):
    output = ""
    for i in range(len(instructions)):
        output += f"{i+1}. {instructions[i]}\n"
    return output
