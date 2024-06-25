import json
import os

import transformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI, OpenAIChat
from langchain.vectorstores import Milvus
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

# LLM
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = OpenAI(model_name="Qwen1.5-14B")
llm_chat = OpenAIChat(model_name="Qwen1.5-14B")
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")
print('Loaded LLM')

# embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
print('Loaded HuggingFaceEmbeddings')

# vector store
db = Milvus(embedding_function=embedding, collection_name="arXiv", connection_args={"host": "172.29.4.47", "port": "19530"})
print('Loaded Milvus')

# Prompt Template
keyword_prompt_template = PromptTemplate(
    template="假设你是一个计算机科学和人工智能领域的专家。给定一个关于计算机科学领域的查询问题：“{user_input}”，请从中提取关键词，至多5个，至少2个，以英文给出。我们更关注其中出现的专业名词，优先关注问题中出现的关键词。请在输出时将关键词用英文逗号 `,` 隔开（形如：`关键词1,关键词2,...`），不要输出其他内容。",
    input_variables=["user_input"]
)
judge_prompt_template = PromptTemplate(
    template="""假设你是一个计算机科学和人工智能领域的专家。给定一个关于计算机科学领域的查询问题，以及从arXiv中检索问题相关关键词得到的论文列表（包括题目、作者和摘要）。请你判断能否据此完整回答该问题。
如果所提供信息能够完整回答该问题，请输出Yes；否则，如果所提供信息不足以完整回答该问题，请输出No。请注意不需要回答该问题，直接输出Yes/No。

查询问题：“{user_input}”
以下是arXiv中的相关论文列表：
{reference}""",
    input_variables=["user_input", "reference"]
)
summary_prompt_template = PromptTemplate(
    template="""假设你是一个计算机科学和人工智能领域的专家。给定一个关于计算机科学领域的查询问题，以及从arXiv中检索问题相关关键词得到的论文列表（包括题目、作者和摘要）。请你根据问题和给出的arXiv论文的摘要回答用户问题。
请参考给出的arXiv论文的摘要，给出完整的、充分的有关该问题的回答，并在回答的文本之后以列表的形式给出解答问题的来源arXiv论文（包括真实的题目和作者）。

查询问题：“{user_input}”
以下是arXiv中的相关论文列表：
{reference}""",
    input_variables=["user_input", "reference"]
)
refind_prompt_template = PromptTemplate(
    template="""假设你是一个计算机科学和人工智能领域的专家。给定一个关于计算机科学领域的查询问题，以及从arXiv中检索问题相关关键词得到的论文列表（包括题目、作者和摘要）。
已经检索到的论文摘要内容不足以回答这个问题。请你根据问题和给出的arXiv论文，给出你认为还需要检索哪些关键词，至多5个，至少2个，以英文给出。我们更关注其中出现的专业名词。请在输出时将关键词用英文逗号 `,` 隔开（形如：`关键词1,关键词2,...`），不要输出其他内容。

查询问题：“{user_input}”
以下是arXiv中的相关论文列表：
{reference}""",
    input_variables=["user_input", "reference"]
)


def calculate_tokens(source):
    return len(tokenizer(source)["input_ids"])


def query_arxiv(keywords):
    answers = {}
    for keyword in keywords:
        if len(keywords) <= 0:
            continue
        search_results = db.similarity_search_with_score(keyword, k=3)
        for result, score in search_results:
            # print(result)
            access_id = result.metadata.get('access_id', None)
            if access_id is None:
                continue
            if access_id in answers.keys():
                answers[access_id]['keywords'].append(keyword)
                answers[access_id]['score'] = max(score, answers[access_id]['score'])
            else:
                answers[access_id] = dict(abstract=result.page_content.replace("\n", "").strip(),
                                          title=result.metadata.get('title', 'No title').replace("\n", "").strip(),
                                          authors=result.metadata.get('authors', 'No authors').replace("\n", "").strip(),
                                          score=score,
                                          keywords=[keyword])
    return answers


def text_ref(ref, detailed=True):
    text = []
    for i, (access_id, item) in enumerate(sorted(ref.items(), key=lambda x: -x[1]['score'])):
        if detailed:
            text.append(f'{i + 1}. {item["title"]} (keywords: {",".join(item["keywords"])})\nAuthors: {item["authors"]}\nAbstract: {item["abstract"]}\n')
        else:
            text.append(f'{i + 1}. {item["title"]} - {item["authors"]}: https://arxiv.org/abs/{access_id}')
    return '\n'.join(text)


def query(question):
    # 拆分关键词
    keyword_prompt = keyword_prompt_template.format(user_input=question)
    keywords = set(map(str.strip, llm_chat(keyword_prompt,  # 生成关键词
                                           temperature=0.3,  # 设置温度
                                           ).strip().lstrip('关键词:').lstrip('关键词：').lstrip('关键词').lstrip('Keywords:').lstrip('keywords:').strip().strip('`').strip().split(',')))
    print(f'拆分关键词: {keywords}')
    final_answer = None
    to_query_keywords = keywords  # 待查询关键词（未查询过的）
    arxiv_reference = {}  # 引用列表
    while final_answer is None or (isinstance(final_answer, str) and len(final_answer) == 0):
        # 查询关键词
        answers = query_arxiv(to_query_keywords)
        keywords.update(to_query_keywords)
        arxiv_reference.update(answers)
        # 使用大模型判断现有内容是否足够
        judge_prompt = judge_prompt_template.format(user_input=question, reference=text_ref(arxiv_reference)).strip()
        tokens = calculate_tokens(judge_prompt)
        if tokens > 8000:  # token限制
            print('token长度超出限制')
            while tokens > 8000:  # 从得分最低的文献开始删，直至满足token限制
                del_key = min(arxiv_reference.keys(), key=lambda k: arxiv_reference[k]['score'])
                print(f'删除了ref: {arxiv_reference[del_key]}')
                del arxiv_reference[del_key]
                judge_prompt = judge_prompt_template.format(user_input=question, reference=text_ref(arxiv_reference)).strip()
                tokens = calculate_tokens(judge_prompt)
        # print(f'LLM输入: {judge_prompt}')
        judge_answer = llm_chat(judge_prompt,
                                temperature=0.2,  # 设置温度
                                max_tokens=50  # 设置最大 token 长度
                                ).strip()
        print(f'LLM响应: {judge_answer}')
        if judge_answer.lower().startswith('yes') or len(to_query_keywords) == 0:  # 如果给出了yes，或者待查列表为空，则生成答案
            summary_prompt = summary_prompt_template.format(user_input=question, reference=text_ref(arxiv_reference)).strip()
            final_answer = llm_chat(summary_prompt,
                                    temperature=0.5,  # 设置温度
                                    ).strip()
        elif judge_answer.lower().startswith('no'):  # 重新确定关键词，未查询过的纳入待查列表
            refind_prompt = refind_prompt_template.format(user_input=question, reference=text_ref(arxiv_reference)).strip()
            refind_answer = llm_chat(refind_prompt,
                                     temperature=0.3,  # 设置温度
                                     ).strip()
            print(f'LLM响应: {refind_answer}')
            new_keywords = set(map(str.strip, refind_answer.lstrip('关键词:').lstrip('关键词：').lstrip('关键词').lstrip('Keywords:').lstrip('keywords:').strip().strip('`').strip().split(',')))
            to_query_keywords = new_keywords - set(keywords)
            print(f'New todo keywords: {to_query_keywords}')
        else:
            print('非法输出，重新调用LLM')
            print(f'LLM响应: {judge_answer}')
    final_answer += f'\n\n参考文献：\n{text_ref(arxiv_reference, False)}'
    print(f'>> Q: {question}')
    print(f'>> A: {final_answer}')
    return final_answer


def answer_questions(input_file, output_file):
    with open(input_file, "r", encoding='utf-8') as infile:
        questions = json.load(infile)
    answers = []
    for question in tqdm(questions):
        question = question['question']
        answer = query(question)
        answers.append({"question": question, "answer": answer})
    with open(output_file, "w", encoding='utf-8') as outfile:
        json.dump(answers, outfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # print(query('什么是大语言模型？'))
    answer_questions("questions.jsonl", "answers.jsonl")
