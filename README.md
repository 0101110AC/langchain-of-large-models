# LangChain with Ollama Local Model

## 项目概述

本项目旨在使用LangChain库和Ollama本地模型来实现一系列自然语言处理任务。LangChain是一个强大的工具，可以帮助开发者快速构建复杂的语言模型应用。Ollama则提供了一个高效的本地模型解决方案，使得我们能够在本地环境中运行和测试模型。

## 环境要求

- Python 3.11.10
- LangChain
- Ollama
- 其他依赖库（根据具体需求安装）

## 安装步骤

1. **安装Python**:
   确保你的系统上已经安装了Python 3.11.10。你可以从[Python官方网站](https://www.python.org/downloads/)下载并安装。

2. **安装依赖库**:
   使用pip安装项目所需的依赖库。
   ```bash
   pip install langchain ollama
   ```

3. **下载Ollama模型**:
   根据项目需求，下载并配置Ollama模型。你可以从Ollama的官方文档中找到具体的下载和配置指南。

## 项目结构

```
langchain-of-large-models/
├── 提示词.ipynb
├── README.md
└── 其他相关文件
```

- `提示词.ipynb`: 包含了使用LangChain和Ollama模型的示例代码。
- `README.md`: 项目的介绍文档。

## 使用示例

在`提示词.ipynb`中，你可以找到以下示例代码：

1. **创建Ollama实例**:
   ```python
   from langchain.llms import Ollama

   llm = Ollama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
   ```

2. **创建提示模板**:
   ```python
   from langchain.prompts import PromptTemplate

   prompt = PromptTemplate(
       input_variables=["question"],
       template="请回答以下问题：{question}"
   )
   ```

3. **创建LLMChain实例**:
   ```python
   from langchain.chains import LLMChain

   chain = LLMChain(llm=llm, prompt=prompt)
   ```

4. **运行链并获取答案**:
   ```python
   question = "如何提高编程技能？"
   answer = chain.run(question)
   print(answer)
   ```

## 贡献

欢迎任何形式的贡献！如果你有任何问题或建议，请在[Issues](https://github.com/your-repo/issues)中提出。如果你想要贡献代码，请先联系本人（3324974030@qq.com)


