from utils import get_content_between_a_b, get_api_response
import torch

import random

from sentence_transformers import  util


class RecurrentGPT:

    def __init__(self, input, short_memory, long_memory, memory_index, embedder):
        self.input = input
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.embedder = embedder
        if self.long_memory and not memory_index:
            self.memory_index = self.embedder.encode(
                self.long_memory, convert_to_tensor=True)
        self.output = {}

    def prepare_input(self, new_character_prob=0.1, top_k=2):

        input_paragraph = self.input["output_paragraph"]
        input_instruction = self.input["output_instruction"]

        instruction_embedding = self.embedder.encode(
            input_instruction, convert_to_tensor=True)

        # get the top 3 most similar paragraphs from memory

        memory_scores = util.cos_sim(
            instruction_embedding, self.memory_index)[0]
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [self.long_memory[idx] for idx in top_k_idx]
        # combine the top 3 paragraphs
        input_long_term_memory = '\n'.join(
            [f"Related Paragraphs {i+1} :" + selected_memory for i, selected_memory in enumerate(top_k_memory)])
        # randomly decide if a new character should be introduced
        if random.random() < new_character_prob:
            new_character_prompt = f"If it is reasonable, you can introduce a new character in the output paragrah and add it into the memory."
        else:
            new_character_prompt = ""

        input_text = f"""I need you to help me write a Galgame Script. Now I give you a memory (a brief summary) of 400 words, you should use it to store the key content of what has been written so that you can keep track of very long context. For each time, I will give you your current memory (a brief summary of previous stories. You should use it to store the key content of what has been written so that you can keep track of very long context), the previously written paragraph, and instructions on what to write in the next paragraph. 
    I need you to write:
    1. Output Paragraph: the next paragraph of the Galgame Script. The output paragraph should contain around 50 sentences and should follow the input instructions.
    2. Output Memory: The updated memory. You should first explain which sentences in the input memory are no longer necessary and why, and then explain what needs to be added into the memory and why. After that you should write the updated memory. The updated memory should be similar to the input memory except the parts you previously thought that should be deleted or added. The updated memory should only store key information. The updated memory should never exceed 80 sentences!
    3. Output Instruction:  instructions of what to write next (after what you have written). You should output 3 different instructions, each is a possible interesting continuation of the story. Each output instruction should contain around 5 sentences 请不要写为什么要这样写的原因，只写下一步要写什么。
    4.非常重要！请在输出信息中除了规定的格式之外全部使用中文（除了人名），注意要符合中文母语的语法和用词习惯。
    Galgame的剧情都非常非常长，一般需要20~30个小时才能读完，动辄几十上百万字。
    你的总体剧情应该发展比较缓慢，充满悬念和节外生枝，剧情发展缓慢的同时，不能平淡无聊，增加较多的小插曲是很好的选择，也可以用来更好的刻画人物性格。
    也就是说，你以一个固定的角色（主角）（一般是男性）视角写故事，你的故事情节中只能出现对话（占绝大部分篇幅）、环境描写（在每一行环境描写使用<environment>标签包裹）和主角的心理活动（在每一行主角的心理活动使用<hearty>标签包裹）。
    注意一定要大量的角色语言。
    角色的语言要富有个性，生动灵活，不要过于死板。
    每行角色语言这样写:”<say><character>某人</character>xxxx</say>“也就是说，每句语言需要用<say>标签包裹，而说这句话的角色名字要用<character>标签包裹。
    每一句角色语言、环境描写、心理描写都要单独一行，单独加上标签，类似一个XML剧本。每个角色都要有自己的名字，不要用“ABC”“甲乙丙”"同学A同学B"这种符号化代称，也不要使用“班主任”“xx领导”等等代称，一切角色在该角色语言的冒号前面用完整的原名来指代，想表达这个角色的身份的话可以在他第一次出场或者其他时候用一两句话自我介绍或其他方式介绍。
    Here are the inputs: 

    Input Memory:  （这是当前的记忆，请参考里面提供的信息撰写具体剧情）
    {self.short_memory}

    Input Paragraph:（这是前面最近的几个段落）
    {input_paragraph}

    Input Instruction:（这是你要写的内容的提纲，请按照提纲写下面的内容）
    {input_instruction}

    Input Related Paragraphs:（这是全部上文中很可能比较相关的一些段落，可以参考里面的信息来提高剧情准确性）
    {input_long_term_memory}
    
    Now start writing, organize your output by strictly following the output format as below:
    Output Paragraph: 
    <string of output paragraph>, around 50-100 sentences.

    Output Memory: 
    Rational: <string that explain how to update the memory>;
    Updated Memory: <string of updated memory>, around 50 sentences

    Output Instruction: 
    Instruction 1: <content for instruction 1>, around 5 sentences
    Instruction 2: <content for instruction 2>, around 5 sentences
    Instruction 3: <content for instruction 3>, around 5 sentences

    Very important!! The updated memory should only store key information. The updated memory should never contain over 500 words!
    Finally, remember that you are writing a novel. Write like a novelist and do not move too fast when writing the output instructions for the next paragraph. Remember that the chapter will contain over 10 paragraphs and the novel will contain over 100 chapters. And this is just the beginning. Just write some interesting staffs that will happen next. Also, think about what plot can be attractive for common readers when writing output instructions. 

    Very Important: 
    You should first explain which sentences in the input memory are no longer necessary and why, and then explain what needs to be added into the memory and why. After that, you start rewrite the input memory to get the updated memory. 
    非常重要！请在输出信息中除了规定的格式之外全部使用中文（除了人名），注意要符合中文母语的语法和用词习惯。
    Galgame的剧情都非常非常长，一般需要20~30个小时才能读完，动辄几十上百万字。
    你的总体剧情应该发展比较缓慢，充满悬念和节外生枝，剧情发展缓慢的同时，不能平淡无聊，增加较多的小插曲是很好的选择，也可以用来更好的刻画人物性格。
    也就是说，你以一个固定的角色（主角）（一般是男性）视角写故事，你的故事情节中只能出现对话（占绝大部分篇幅）、环境描写（在每一行环境描写使用<environment>标签包裹）和主角的心理活动（在每一行主角的心理活动使用<hearty>标签包裹）。
    注意一定要大量的角色语言。
    角色的语言要富有个性，生动灵活，不要过于死板。
    每行角色语言这样写:”<say><character>某人</character>xxxx</say>“也就是说，每句语言需要用<say>标签包裹，而说这句话的角色名字要用<character>标签包裹。
    每一句角色语言、环境描写、心理描写都要单独一行，单独加上标签，类似一个XML剧本。
    每个角色都要有自己的名字，不要用“ABC”“甲乙丙”"同学A同学B"这种符号化代称，也不要使用“班主任”“xx领导”等等代称，一切角色在该角色语言的冒号前面用完整的原名来指代，想表达这个角色的身份的话可以在他第一次出场或者其他时候用一两句话自我介绍或其他方式介绍。
    {new_character_prompt}
    """
        return input_text

    def parse_output(self, output):
        try:
            output_paragraph = get_content_between_a_b(
                'Output Paragraph:', 'Output Memory', output)
            output_memory_updated = get_content_between_a_b(
                'Updated Memory:', 'Output Instruction:', output)
            self.short_memory = output_memory_updated
            ins_1 = get_content_between_a_b(
                'Instruction 1:', 'Instruction 2', output)
            ins_2 = get_content_between_a_b(
                'Instruction 2:', 'Instruction 3', output)
            lines = output.splitlines()
            # content of Instruction 3 may be in the same line with I3 or in the next line
            if lines[-1] != '\n' and lines[-1].startswith('Instruction 3'):
                ins_3 = lines[-1][len("Instruction 3:"):]
            elif lines[-1] != '\n':
                ins_3 = lines[-1]

            output_instructions = [ins_1, ins_2, ins_3]
            assert len(output_instructions) == 3

            output = {
                "input_paragraph": self.input["output_paragraph"],
                "output_memory": output_memory_updated,  # feed to human
                "output_paragraph": output_paragraph,
                "output_instruction": [instruction.strip() for instruction in output_instructions]
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
                f.write(f"Writer's output here:\n{response}\n\n")

        self.long_memory.append(self.input["output_paragraph"]+self.output["output_paragraph"])
        self.memory_index = self.embedder.encode(
            self.long_memory, convert_to_tensor=True)
