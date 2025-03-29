# LangChain 核心组件学习笔记

## 前言

LangChain是一个强大的框架，专为开发基于大语言模型(LLM)的应用而设计。它提供了一系列组件和工具，帮助开发者构建复杂的AI应用，如聊天机器人、问答系统、文档分析等。本笔记记录了LangChain核心组件的学习过程和关键概念。

## 一、提示词工程（Prompt Engineering）

提示词工程是与大语言模型交互的基础，它关注如何构建有效的提示以获得期望的输出。LangChain提供了多种提示模板和选择器，使提示词工程更加系统化。

### 什么是Few-Shot Learning？

Few-Shot Learning（少样本学习）是一种提示技术，通过向模型提供少量示例来引导其理解任务模式。这种方法特别适用于复杂或专业领域的任务，可以显著提高模型的表现。

### 语义相似度选择器实现

语义相似度选择器是Few-Shot Learning的高级实现，它能根据输入问题自动选择最相关的示例，而不是使用固定示例。这种动态选择方式可以大幅提高模型回答的准确性。

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 初始化本地embedding模型（此处使用预训练好的MiniLM模型）
model_name = "D:/33249/代码/python/大模型/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 创建语义相似度选择器
input_keys = ["question"]  # 指定输入键用于相似度计算

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, 
    embeddings, 
    Chroma, 
    k=1,
    input_keys=input_keys
)
```

**参数说明：**
- `examples`: 示例数据集（需提前定义），包含问题和答案对
- `embeddings`: 本地加载的向量编码模型，用于计算语义相似度
- `Chroma`: 选择使用的向量数据库实现，用于高效存储和检索向量
- `k=1`: 返回Top1最相似示例，可根据需要调整
- `input_keys`: 指定用于相似度计算的输入字段（本例使用question字段）

### Few-Shot提示模板配置

一旦有了语义选择器，我们需要创建一个Few-Shot提示模板来组织示例和用户输入：

```python
# 完整提示模板配置
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["question", "answer"],
        template="问题：{question}\n{answer}"
    ),
    suffix="问题：{question}",
    input_variables=["question"]
)
```

**配置说明：**
1. `example_selector`: 绑定语义相似度选择器，用于动态选择相关示例
2. `example_prompt`: 定义单个示例的展示格式，指定如何呈现问题和答案
3. `suffix`: 设置问题输入模板，在所有示例之后添加用户的实际问题
4. `input_variables`: 声明模板变量，指定哪些变量需要在运行时提供

这种动态Few-Shot方法比静态示例更灵活，能够根据用户输入自适应地选择最相关的示例，提高模型回答的质量和相关性。

## 二、提示模板与解析器（Prompts & Parsers）

LangChain提供了强大的提示模板系统和输出解析器，使开发者能够标准化与模型的交互并处理返回结果。

### 提示模板的作用

提示模板（PromptTemplate）是LangChain的核心概念之一，它允许开发者创建结构化的提示，包含固定文本和动态变量。这种模板化方法有几个关键优势：

1. **一致性**：确保每次请求使用相同的提示结构
2. **可复用性**：模板可以在不同场景中重复使用
3. **可维护性**：集中管理提示，便于更新和优化

### PromptTemplate 配置

```python
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import OllamaLLM

# 初始化本地Ollama模型
model = OllamaLLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.1)
```

**温度参数说明：**
- `temperature=0.1` 控制生成结果的创造性
- 值越低输出越保守和确定性，值越高输出越随机和创造性
- 需要重新实例化模型才能生效

```python
# 定义一个HumanMessagePromptTemplate
human_message_prompt = HumanMessagePromptTemplate.from_template("{input_text}")

# 使用消息模板列表初始化ChatPromptTemplate
chat = ChatPromptTemplate.from_messages([human_message_prompt])

# 创建更复杂的模板
template_string = """Translate the text \
    that is delimited by triple backticks \
    into a style that is {style}.\
    text: ```{input_text}```"""

prompt_template = ChatPromptTemplate.from_template(template_string)
# 查看模板的输入变量
print(prompt_template.messages[0].prompt.input_variables)  # 输出: ['input_text', 'style']
```

### 格式化消息与模型调用

创建模板后，我们需要用实际值填充模板并调用模型：

```python
# 准备输入数据
customer_style = """ American English\
    in a calm and respectful tone\
    """

customer_email = """Arrr,I be fuming that me blender lid\
    flew off and splattered me kitchen walls\
    with smoothie! And to make matters worse,\
    the warranty don't cover the cost of\
    cleaning up me kitchen walls. I need yer help\
    right now, matey!
    """
    
# 格式化消息
customer_messages = prompt_template.format_messages(
    style = customer_style,
    input_text = customer_email
)

# 调用模型获取响应
customer_response = model.invoke(customer_messages)
```

**输出结果示例：**
```
Hello! I'm frustrated because my blender lid flew off and splattered smoothie all over my kitchen walls. 
Unfortunately, the warranty doesn't cover cleaning costs. I would really appreciate your assistance as soon as possible.
```

这个例子展示了如何使用提示模板将海盗风格的抱怨转换为礼貌的美式英语，展示了模板在文本风格转换中的应用。

### 输出解析器

输出解析器是LangChain的另一个重要组件，它们将模型的原始文本输出转换为结构化数据，便于程序处理。

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# 字符串解析器（最简单的解析器）
parser = StrOutputParser()

# 定义JSON输出结构
class ReviewData(BaseModel):
    things: str = Field(description="产品名称")
    gift: bool = Field(description="是否是礼物")
    delivery_days: int = Field(description="配送天数")
    price_value: Optional[str] = Field(description="价格评价")

# 创建JSON解析器
json_parser = JsonOutputParser(pydantic_object=ReviewData)
```

**解析器类型说明：**
- `StrOutputParser`: 最基本的解析器，直接返回模型输出的字符串
- `JsonOutputParser`: 将模型输出解析为JSON对象，需要定义Pydantic模型指定结构
- 其他解析器：LangChain还提供了列表解析器、枚举解析器等多种专用解析器

```python
# 构建提示模板
review_template = """\
    For the following text, extract the following information:
    things: what is the product?
    gift: is it a gift?
    delivery_days: how many days does it take to deliver?
    price_value: any comments on the price?
    
    Format the output as JSON with the following keys:
    things, gift, delivery_days, price_value
    
    text: {text}
    """

# 创建完整处理链
chain = (
    ChatPromptTemplate.from_template(review_template) 
    | model 
    | json_parser
)

# 调用链处理输入
result = chain.invoke({"text": "This leaf blower is pretty affordable and does the job well. I would buy another one for my friend who needs a leaf blower. It will take 2 days to arrive."})
```

**输出结果示例：**
```json
{
    "things": "leaf blower",
    "gift": true,
    "delivery_days": 2,
    "price_value": "pretty affordable"
}
```

这个例子展示了如何从非结构化文本中提取结构化信息，这在电商评论分析、客户反馈处理等场景中非常有用。

## 三、工作流编排（Workflow Orchestration）

工作流编排是LangChain的核心功能之一，它允许开发者将多个组件连接成处理流程，实现复杂的任务处理。

### Runnable协议与链式调用

LangChain的Runnable协议是一种统一接口，允许不同组件（如模型、提示模板、解析器等）通过管道操作符（|）连接在一起，形成处理链。这种设计使得工作流构建变得直观且灵活。

### 流式处理（Streaming）

流式处理允许模型生成的内容逐块返回，而不是等待完整响应，这对于构建交互式应用非常重要。

```python
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate

# 初始化本地Ollama模型
model = OllamaLLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434")

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("给我讲一个关于{topic}的故事")
])

# 构建处理链
parser = StrOutputParser()
chain = prompt | model | parser

# 流式输出示例
for chunk in model.stream("天空是什么颜色"):
    print(chunk, end="|", flush=True)
```

**流式输出的优势：**
1. **用户体验**：用户可以看到实时生成的内容，减少等待感
2. **早期反馈**：开发者可以在完整响应前获取部分输出，便于调试
3. **资源效率**：可以在生成过程中处理数据，而不必等待全部完成

### 异步调用与多线程处理

对于需要处理多个请求或执行并行任务的应用，LangChain提供了异步API支持：

```python
import asyncio
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate

# 包装生成器为异步可迭代对象的类
class AsyncGeneratorWrapper:
    def __init__(self, gen):
        # 初始化时传入一个生成器对象
        self.gen = gen

    def __aiter__(self):
        # 使该类的实例成为异步可迭代对象
        return self

    async def __anext__(self):
        try:
            # 使用 asyncio.to_thread 避免阻塞事件循环
            return await asyncio.to_thread(next, self.gen)
        except StopIteration:
            # 当生成器耗尽时，抛出异步迭代结束的异常
            raise StopAsyncIteration

# 创建异步流式处理函数
async def async_stream():
    # 调用链式处理流程的 stream 方法，传入 topic 参数，得到一个生成器
    gen = chain.stream({"topic": "天空"})
    # 将生成器包装成异步可迭代对象
    async_gen = AsyncGeneratorWrapper(gen)
    # 异步迭代生成器，逐块输出内容
    async for chunk in async_gen:
        print(chunk, end="|", flush=True)

# 多线程并发调用示例
async def main():
    # 并发执行多个异步任务
    await asyncio.gather(async_stream1(), async_stream2())
```

**异步处理的关键优势：**
- **并发处理**：同时处理多个请求，提高吞吐量
- **资源利用**：在等待I/O操作（如API调用）时可以执行其他任务
- **响应性**：保持应用的响应能力，避免阻塞主线程

### JSON输出解析与事件流

LangChain支持结构化数据的流式处理和事件驱动编程模型：

```python
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
import asyncio

# 初始化模型
model = OllamaLLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
# 创建JSON解析链
chain = (model | JsonOutputParser())

# 流式处理JSON输出示例
async def async_stream():
    # 调用chain的stream方法，传入请求内容
    gen = chain.stream(
        "以JSON格式输出法国、西班牙和日本的国家及人口列表。"
        '使用一个带有"countries"外部键的字典，其中包含国家列表。'
        '每个国家都应该有键"name"和"population"。'
    )
    # 将同步生成器包装成异步生成器
    async_gen = AsyncGeneratorWrapper(gen)
    # 异步迭代异步生成器，逐块获取模型输出
    async for text in async_gen:
        print(text, flush=True)
```

**JSON流式处理的应用场景：**
- **实时数据可视化**：逐步构建和更新数据可视化
- **增量数据处理**：在接收完整数据前开始处理部分数据
- **长时间运行的任务**：提供任务进度的实时反馈

```python
# 事件流处理示例
async def event_stream():
    # 初始化一个空列表，用于存储模型返回的事件
    events = []
    # 异步迭代模型的事件流
    async for event in model.astream_events(input="hello", version="v2"):
        # 将每个事件添加到events列表中
        events.append(event)
    # 打印存储所有事件的列表
    print(events)
```

**事件流的优势：**
- **细粒度控制**：可以响应生成过程中的各种事件（开始、流式输出、结束等）
- **自定义处理**：为不同类型的事件实现不同的处理逻辑
- **监控与日志**：记录模型调用的详细信息，便于调试和性能分析

## 四、最佳实践与注意事项

在使用LangChain开发应用时，以下是一些重要的最佳实践：

1. **提示模板设计**：
   - 保持提示清晰、具体且结构化
   - 使用Few-Shot示例引导模型理解任务
   - 为复杂任务提供步骤分解

2. **错误处理**：
   - 实现重试机制处理临时性错误
   - 设置超时防止长时间等待
   - 验证模型输出，处理格式错误

3. **性能优化**：
   - 使用流式处理提高响应速度
   - 实现缓存减少重复调用
   - 对于批量任务，使用异步API

4. **安全考虑**：
   - 实施输入验证防止提示注入
   - 限制模型输出长度避免资源耗尽
   - 审查敏感信息，避免在提示中包含私密数据

## 持续更新中...

本笔记将随着学习的深入不断更新，后续将添加记忆组件、代理系统、工具集成等更多LangChain核心功能的学习内容。