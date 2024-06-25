# RAG实验(必做)

## 题目要求

### 基础要求

- 构建一个针对arXiv的知识问答系统
- 要求如下
    - 给定一个入口，用户可以输入提问
    - 不要求要求构建GUI界面
    - 用户通过对话进行交互
    - 系统寻找与问题相关的论文abstract
    - 使用用户的请求对向量数据库进行请求
    - 寻找与问题最为相关的abstract
    - 系统根据问题和论文abstract回答用户问题，并给出解答问题的信息来源
- 示例

### 进阶要求

- 提示优化
    - 用户给出的问题或陈述不一定能够匹配向量数据库的查询
    - 使用大模型对用户的输入进行润色，提高找到对应文档的概率
    - 思路提示(解决思路不唯一，提示仅作为可能的思路示例)
        - 观察不同输入后向量数据库找到对应文档的概率
        - 总结适用于查询的语句
        - 构建提示(prompt)实现对用户输入的润色
    - 查询迭代
        - 单次的查询可能无法寻找到用户所期望的答案
        - 需要通过多轮的搜索和尝试才能获得较为准确的答案
        - 思路提示
        - 如何将用户的需求拆解，变成可以拆解的逻辑步骤
        - 如何判断已经获得准确的答案并停止迭代
        - 如何再思路偏移后进行修正

## 资源简介

### 大模型(Qwen1.5-14B)

Qwen1.5-14B模型已接入LangChain Openai API,调用示例如下

``` python
from langchain.llms import OpenAI, OpenAIChat
import os
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = OpenAI(model_name="Qwen1.5-14B")
llm_chat = OpenAIChat(model_name="Qwen1.5-14B")
```

openai包使用特定版本，避免与langchain不兼容 pip install openai==0.28

### 嵌入模型(sentence-transformers/all-MiniLM-L12-v2)

* 嵌入模型使用huggingface中的all-MiniLM-L12-v2模型

``` python
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
```

当前由于huggingface被墙，无梯子可以使用镜像，详见 https://hf-mirror.com/

### 向量数据库

arXiv数据存在Milvus中

``` python
from langchain.vectorstores import Milvus
db = Milvus(embedding_function=embedding, collection_name="arXiv",connection_args={"host": "172.29.4.47", "port": "19530"})
```

由于向量数据库与SDK存在强绑定关系，安装milvus包时请检查版本： pip install pymilvus==2.2.6

#### 数据项解释

- vector： 论文abstract的向量化表示
- access_id：论文的唯一id
    - https://arxiv.org/abs/{access_id} 论文的详情页
    - https://arxiv.org/pdf/{access_id} 论文的pdf地址
- authors：论文的作者
- title：论文的题目
- comments：论文的评论，一般为作者的补充信息
- journal_ref：论文的发布信息
- doi：电子发行doi
- text：论文的abstract (为了兼容langchain必须命名为text)
- categories：论文的分类

### LangChain

- langchain官方文档 https://python.langchain.com/
- langchain官方课程 https://learn.deeplearning.ai/langchain

## 提交内容

1. 代码实现
2. 预置题目
    1. 预置由json文件构成，包含10个问题(question项目)
    2. 使用算法回答其中问题，答案存在answer项内
    3. 该文件存储为answer.json，单独提交

## 实现思路

先做关键词提取，然后到向量数据库检索文献，交给LLM判断现有的文献是否足够，如果不足够就继续生成关键词，直到LLM认为足够或者没有生成新的关键词了，最后总结输出（按文献与关键词的相似度排列给出参考文献）。
对于可能超出token限制的场景，逐个删去相似度最低的文献，直至满足token限制。
