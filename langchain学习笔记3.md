# LangChain 核心组件学习笔记（三）

## 前言

继上一篇笔记后，本文将继续探索LangChain框架的核心组件，重点关注基于文档的问答、评估系统和代理机制。这些组件是构建复杂AI应用的关键部分，能够帮助开发者实现更智能、更可靠的大语言模型应用。

## 四、基于文档的问答（Retrieval-based QA）

基于文档的问答是LangChain的核心应用场景之一，它允许大语言模型基于特定文档集合回答问题，而不仅仅依赖于模型的预训练知识。

### 向量存储与检索

向量存储是实现文档检索的基础，它将文本转换为向量表示，并支持高效的相似度搜索。

```python
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import HuggingFaceEmbeddings

# 初始化本地Ollama模型
model = OllamaLLM(model="deepseek-r1:1.5b",
                  base_url="http://localhost:11434",temperature=0.5)

# 加载数据文件
file = 'OutdoorClothingCatalog_1000.csv'
data = CSVLoader(file_path=file,encoding='utf-8')

# 创建嵌入模型
local_model_path = "all-MiniLM-L6-v2" 
embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path,
    model_kwargs={"device": "cuda"}  # 如需用GPU，改为"cuda"
)

# 创建向量存储索引
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings
).from_loaders([data])
```

**向量存储的关键组件：**
1. **文档加载器**：从不同来源（CSV、PDF、网页等）加载文档
2. **嵌入模型**：将文本转换为向量表示
3. **向量数据库**：存储和检索向量化的文档

### 构建检索问答链

检索问答链将文档检索与语言模型结合，实现基于文档的智能问答：

```python
# 创建一个RetrievalQA对象
qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={"document_separator":"<<<<>>>>"}
)

# 执行查询
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = index.query(query, llm=model)
```

**检索问答链的工作原理：**
1. 将用户问题转换为向量表示
2. 在向量数据库中检索相似文档
3. 将检索到的文档与原始问题一起发送给语言模型
4. 语言模型基于检索到的上下文生成回答

## 五、评估系统（Evaluation）

评估系统是确保LLM应用质量的关键组件，它提供了一系列工具来评估模型输出的质量、准确性和相关性。

### 问答评估

问答评估关注模型回答问题的能力，包括准确性、完整性和相关性：

```python
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAGenerateChain
from langchain.chat_models import ChatOllama
from langchain.schema import BaseOutputParser

# 自定义提示模板
prompt_template = """ 
Given the following English product description, generate one English Question-Answer pair (QA pair) that:
1. Focuses on specific data in Specs, Construction, or Features (e.g., weight, material, dimensions).
2. The question ends with a question mark (?) and the answer is an exact match from the description.

Strict format:
Question: <specific English question>
Answer: <exact English answer>

Product description:
{doc}
"""

# 自定义输出解析器
class QAOutputParser(BaseOutputParser):
    def parse(self, text):
        cleaned_text = "\n".join([line for line in text.split('\n') if "Question:" in line or "Answer:" in line])
        question = ""
        answer = ""
        for line in cleaned_text.split('\n'):
            line = line.strip()
            if line.startswith("Question:"):
                question = line[len("Question:"):].strip()
                if not question.endswith('?'):
                    question += '?'
            elif line.startswith("Answer:"):
                answer = line[len("Answer:"):].strip()
        return {"question": question, "answer": answer}

# 初始化模型和问答生成链
llm = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.0
)
qa_chain = QAGenerateChain(llm=llm, prompt=prompt)
qa_chain.output_parser = QAOutputParser()
```

**评估系统的关键技术：**
1. **自定义提示模板**：引导模型生成特定格式的问答对
2. **输出解析器**：将模型输出解析为结构化数据
3. **评估指标**：准确率、召回率、F1分数等

### 评估挑战与解决方案

在使用本地模型进行评估时，常见的挑战包括：

1. **输出解析器不匹配**：当使用非OpenAI模型时，模型输出格式可能与预期不符
2. **未显式配置提示模板**：需要显式定义强制结构化输出的提示模板

解决方案：
- 自定义输出解析器，适应不同模型的输出格式
- 设计明确的提示模板，引导模型生成结构化输出
- 使用温度参数（temperature=0.0）减少输出的随机性

## 六、代理系统（Agent）

代理系统是LangChain最强大的功能之一，它允许语言模型使用工具、执行推理并采取行动来解决复杂问题。

### 代理类型与工具

LangChain提供了多种代理类型，适用于不同的应用场景：

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

# 初始化模型
llm = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.0
)

# 加载工具
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# 初始化代理
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    has_extended_color_support=True,
    handle_parsing_errors=True,
    verbose=True
)
```

**代理类型说明：**
- `AgentType.ZERO_SHOT_REACT_DESCRIPTION`：零样本学习代理，能够根据工具描述决定使用哪种工具
- `AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION`：专为聊天场景设计的零样本代理
- `AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION`：具有对话记忆功能的聊天代理

### 自定义代理实现

对于更复杂的应用场景，可以自定义代理的行为和输出解析：

```python
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re
from langchain.schema import AgentAction, AgentFinish

# 自定义输出解析器
class CustomOutputParser(AgentOutputParser):
    def parse(self, text):
        match = re.search(r'({.*})', text, re.DOTALL)
        if match:
            json_output = match.group(1)
            try:
                import json
                data = json.loads(json_output)
                return AgentFinish(
                    return_values={"output": data},
                    log=text
                )
            except json.JSONDecodeError:
                pass
        raise OutputParserException(f"无法解析输出: {text}")

# 自定义提示模板
template = """{input}
{intermediate_steps}
{memory_key}"""
memory_key = "chat_history"

# 配置内存
memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True, output_key='output')

# 初始化代理
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", memory_key],
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=tool_names,
    memory=memory,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
```

**代理系统的核心组件：**
1. **工具集**：代理可以使用的外部功能（如计算器、搜索引擎等）
2. **输出解析器**：解析代理的决策和行动
3. **记忆系统**：存储对话历史和中间状态
4. **执行器**：协调代理、工具和记忆的交互

## 七、最佳实践

### 本地模型优化
1. 使用较小的模型（如deepseek-r1:1.5b）进行快速原型开发
2. 对于复杂任务，考虑使用更大的模型或模型量化技术
3. 设置适当的温度参数（temperature）控制输出的随机性

### 代理系统设计
1. 为代理提供清晰的工具描述和使用示例
2. 实现自定义输出解析器处理不同模型的输出格式
3. 添加错误处理机制（handle_parsing_errors=True）

### 评估系统实施
1. 创建多样化的测试集，覆盖不同类型的查询
2. 使用自定义评估指标衡量系统性能
3. 实现人类反馈循环，持续改进系统质量

## 持续更新中...

本笔记将随着学习的深入不断更新，后续将添加更多LangChain高级功能的学习内容，如多模态应用、长文本处理策略等。
