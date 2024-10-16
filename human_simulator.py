from utils import get_content_between_a_b, parse_instructions,get_api_response

class Human:

    def __init__(self, input, memory, embedder):
        self.input = input
        if memory:
            self.memory = memory
        else:
            self.memory = self.input['output_memory']
        self.embedder = embedder
        self.output = {}


    def prepare_input(self):
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        user_edited_plan = self.input["output_instruction"]

        input_text = f"""
        Now imagine you are a novelist writing a Chinese Galgame Script with the help of ChatGPT. You will be given a previously written paragraph (wrote by you), and a paragraph written by your ChatGPT assistant, a summary of the main storyline maintained by your ChatGPT assistant, and a plan of what to write next proposed by your ChatGPT assistant.
    I need you to write:
    1. Extended Paragraph: Extend the new paragraph written by the ChatGPT assistant to twice the length of the paragraph written by your ChatGPT assistant.
    2. Selected Plan: Just copy the plan proposed by your ChatGPT assistant without any adjustment.
    3. Revised Plan: Revise the selected plan into an outline of coming next paragraphs.
    
    Previously written paragraph:  （这是前面的段落，请确保段落前后连贯，衔接不要过于突兀，当然有时候也可以用几句话快速切换场景）
    {previous_paragraph}

    The summary of the main storyline maintained by your ChatGPT assistant:（这是当前的“记忆”或者说是相关信息，请参考里面提供的信息撰写具体剧情）
    {memory}

    The new paragraph written by your ChatGPT assistant:
    {writer_new_paragraph}

    The plan of what to write next proposed by your ChatGPT assistant:
    {user_edited_plan}

    Now start writing, organize your output by strictly following the output format as below,所有输出仍然保持是中文:
    
    Extended Paragraph: 
    <string of output paragraph>, 写的越长越好，但是不能有废话，要紧扣主题，不要离题。至少60句话。可以超过300句话，没有上限。

    Selected Plan: 
    <copy the plan here>

    Revised Plan:（注意你在写当前剧情的时候也要注意与后面剧情的衔接问题）
    <string of revised plan>, keep it short, around 13-25 sentences.Write them in a row,avoid any r"\n".

    Very Important:
    Remember that you are writing a novel. Write like a novelist and do not move too fast when writing the plan for the next paragraph. Think about how the plan can be attractive for common readers when selecting and extending the plan. Remember to follow the length constraints! Remember that the chapter will contain over 10 paragraphs and the novel will contain over 100 chapters. And the next paragraph will be the second paragraph of the second chapter. You need to leave space for future stories.
    非常重要！请在输出信息中除了规定的格式之外全部使用中文（除了人名），注意要符合中文母语的语法和用词习惯。
    Galgame的剧情都非常非常长，一般需要20~30个小时才能读完，动辄几十上百万字。
    你的总体剧情应该发展比较缓慢，充满悬念和节外生枝，剧情发展缓慢的同时，不能平淡无聊，增加较多的小插曲是很好的选择，也可以用来更好的刻画人物性格。
    也就是说，你以一个固定的角色（主角）（一般是男性）视角写故事，你的故事情节中只能出现对话（占绝大部分篇幅）、环境描写（在每一行环境描写使用<environment>标签包裹）和主角的心理活动（在每一行主角的心理活动使用<hearty>标签包裹）。
    注意一定要大量的角色语言。
    角色的语言要富有个性，生动灵活，不要过于死板。
    每行角色语言这样写:”<say><character>某人</character>xxxx</say>“也就是说，每句语言需要用<say>标签包裹，而说这句话的角色名字要用<character>标签包裹。
    每一句角色语言、环境描写、心理描写都要单独一行，单独加上标签，类似一个XML剧本。每个角色都要有自己的名字，不要用“ABC”“甲乙丙”"同学A同学B"这种符号化代称，也不要使用“班主任”“xx领导”等等代称，一切角色在该角色语言的冒号前面用完整的原名来指代，想表达这个角色的身份的话可以在他第一次出场或者其他时候用一两句话自我介绍或其他方式介绍。
    
    """
        return input_text
    
    def parse_plan(self,response):
        plan = get_content_between_a_b('Selected Plan:','Reason',response)
        return plan


    def select_plan(self,response_file="response.txt"):
        
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        previous_plans = self.input["output_instruction"]
        prompt = f"""
    Now imagine you are a helpful assistant that help a novelist with decision making. You will be given a previously written paragraph and a paragraph written by a ChatGPT writing assistant, a summary of the main storyline maintained by the ChatGPT assistant, and 3 different possible plans of what to write next.
    I need you to:
    Select the most interesting and suitable plan proposed by the ChatGPT assistant.

    Previously written paragraph:  （这是前面的段落，请确保段落前后连贯，衔接不要过于突兀）
    {previous_paragraph}

    The summary of the main storyline maintained by your ChatGPT assistant:（这是当前的“记忆”或者说是相关信息，请参考里面提供的信息）
    {memory}

    The new paragraph written by your ChatGPT assistant:
    {writer_new_paragraph}

    Three plans of what to write next proposed by your ChatGPT assistant:
    {parse_instructions(previous_plans)}

    Now start choosing, organize your output by strictly following the output format as below:
      
    Selected Plan: 
    <copy the selected plan here>

    Reason:
    <Explain why you choose the plan>
    """
        print(prompt+'\n'+'\n')

        response = get_api_response(prompt)

        plan = self.parse_plan(response)
        while plan == None:
            response = get_api_response(prompt)
            plan= self.parse_plan(response)

        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Selected plan here:\n{response}\n\n")

        return plan
        
    def parse_output(self, text):
        try:
            if text.splitlines()[0].startswith('Extended Paragraph'):
                new_paragraph = get_content_between_a_b(
                    'Extended Paragraph:', 'Selected Plan', text)
            else:
                new_paragraph = text.splitlines()[0]

            lines = text.splitlines()
            if lines[-1] != '\n' and lines[-1].startswith('Revised Plan:'):
                revised_plan = lines[-1][len("Revised Plan:"):]
            elif lines[-1] != '\n':
                revised_plan = lines[-1]

            output = {
                "output_paragraph": new_paragraph,
                # "selected_plan": selected_plan,
                "output_instruction": revised_plan,
                # "memory":self.input["output_memory"]
            }

            return output
        except:
            return None

    def step(self, response_file="response.txt"):

        prompt = self.prepare_input()
        print(prompt+'\n'+'\n')

        response = get_api_response(prompt)
        self.output = self.parse_output(response)
        while self.output == None:
            response = get_api_response(prompt)
            self.output = self.parse_output(response)
        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Human's output here:\n{response}\n\n")
