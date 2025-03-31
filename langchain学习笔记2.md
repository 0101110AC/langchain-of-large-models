# LangChain 核心组件学习笔记（二）

## 一、链式组件（Chain）

链式组件是LangChain实现复杂工作流的核心机制，通过将多个组件连接形成处理流水线。

### 基础链类型

链式组件的核心设计模式是管道过滤器模式，数据通过标准化的JSON格式在组件间传递。关键技术原理包括：
1. **输入输出映射**：通过output_key指定输出字段，自动匹配后续链的input_variables
2. **执行顺序控制**：SimpleSequentialChain内部维护执行队列，按chains参数顺序执行
3. **调试机制**：verbose模式使用langchain的callback系统记录执行日志

典型数据处理流程：
```mermaid
flow LR
    A[原始输入] --> B(预处理链)
    B --> C{格式校验}
    C -->|通过| D[翻译链]
    C -->|拒绝| E[错误处理]
    D --> F[输出结果]
```

代码示例：

```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate

# 初始化本地模型
model = OllamaLLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434")

# 创建翻译链
translation_prompt = ChatPromptTemplate.from_template(
    "将以下文本翻译成{target_language}:\n{input_text}"
)
translation_chain = LLMChain(
    llm=model,
    prompt=translation_prompt,
    output_key="translated_text"
)

# 创建总结链
summary_prompt = ChatPromptTemplate.from_template(
    "用{length}句话总结以下内容:\n{translated_text}"
)
summary_chain = LLMChain(
    llm=model,
    prompt=summary_prompt,
    output_key="summary"
)

# 组合成顺序链
full_chain = SimpleSequentialChain(
    chains=[translation_chain, summary_chain],
    verbose=True
)
```


**实现原理**：
- 链间通信使用Pydantic模型验证数据格式
- 自动重试机制处理模型调用失败（默认3次重试）
- 内置prompt模板缓存，相同模板哈希值不会重复编译

**注意事项：**
1. 使用SimpleSequentialChain时需确保前序链的输出格式与后续链的输入要求匹配
2. verbose参数开启时可查看链式调用的中间过程

### 复杂工作流示例

```python
from langchain.chains import TransformChain

**TransformChain工作原理**：
- 输入输出验证：通过input_variables/output_variables定义数据契约
- 转换函数执行：transform方法接收字典输入，返回处理后的字典
- 类型安全：使用Pydantic进行运行时类型校验

# 自定义转换链
def length_validator(inputs):
    text = inputs["text"]
    return {"validated_text": text[:1000]}  # 限制输入长度

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["validated_text"],
    transform=length_validator
)

# 构建完整工作流
final_chain = transform_chain | translation_chain | summary_chain

# 调用示例
result = final_chain.invoke({
    "text": "超长输入文本...",
    "target_language": "法语"
})
```

## 二、记忆模块（Memory）

记忆模块使对话系统能够保持上下文，实现多轮对话能力。

### 基础记忆配置

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 初始化记忆模块
memory = ConversationBufferMemory(
    memory_key="chat_history",
    max_token_limit=400,
    return_messages=True
)

# 创建对话链
conversation = ConversationChain(
    llm=model,
    memory=memory,
    verbose=False
)

# 多轮对话示例
conversation.predict(input="你好，我是王小明")
conversation.predict(input="我的名字是什么？")
```


**实现原理**：
- 链间通信使用Pydantic模型验证数据格式
- 自动重试机制处理模型调用失败（默认3次重试）
- 内置prompt模板缓存，相同模板哈希值不会重复编译

**注意事项：**
1. 使用max_token_limit控制记忆长度，避免超出模型上下文限制
2. 遇到LangChainDeprecationWarning时需参考官方迁移指南更新代码

### 记忆优化策略

```python
from langchain.memory import ConversationSummaryBufferMemory
from transformers import AutoTokenizer

# 加载专用分词器
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    trust_remote_code=True
)

# 带摘要的记忆模块
memory = ConversationSummaryBufferMemory(
    llm=model,
    max_token_limit=1000,
    tokenizer=tokenizer,
    human_prefix="用户",
    ai_prefix="助手"
)

# 记忆保存示例
memory.save_context(
    {"input": "AI是什么？"},
    {"output": "人工智能是模拟人类智能的系统"}
)
```

**记忆维护机制**：
1. Token计算：使用transformers分词器精确统计token消耗
2. 滚动窗口：当token数超过max_token_limit时，自动移除最早消息
3. 摘要生成：结合LLM生成对话摘要保留关键信息

**关键配置说明：**
1. 使用专用分词器准确计算token数量
2. human_prefix/ai_prefix参数可自定义对话角色标识
3. 当记忆超过限制时自动生成摘要保持上下文

## 三、最佳实践

### 链式组件优化
1. 为每个链设置明确的输入输出键
2. 使用TransformChain进行数据预处理
3. 为耗时操作添加缓存机制

### 记忆管理策略
1. 根据场景选择记忆类型（Buffer/Summary）
2. 定期清理过时对话记录
3. 重要信息手动保存到长期记忆

本笔记将持续更新，后续将添加代理系统、工具集成等进阶内容。