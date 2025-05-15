# 🦙🔧 Hermes-Function-Calling (CPU Edition) 🚀

> **This repository is a modified fork of [NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling), adapted (mostly by AI 🤖) to run efficiently on CPUs using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)!**

---

## ✨ Features

- 🦙 **Llama 3/2/1 GGUF Model Support** via `llama-cpp-python`
- 🖥️ **CPU-Only**: No GPU required!
- 🔌 **OpenAI-Compatible Function Calling** (Tools/Functions)
- 🧩 **Composable Prompts** and **Multi-Function Chaining**
- 🛠️ **Ready-to-Run Notebooks** for local inference and tool use
- 📝 **Minimal Dependencies** for easy setup

---

## 📦 Installation

```bash
git clone https://github.com/your-username/Hermes-Function-Calling.git
cd Hermes-Function-Calling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🦙 Model Download

- Download a GGUF model (e.g., Hermes 2 Pro, Llama 3) from [HuggingFace](https://huggingface.co/NousResearch) or [Ollama](https://ollama.com/library).
- Place the `.gguf` file in a known location.

---

## 🚀 Usage

### 1️⃣ Run Example Notebooks

Open any notebook in `examples/` (e.g., `lllama-cpp-multiple-fn.ipynb`) in JupyterLab or VSCode.

```bash
jupyter lab
```

### 2️⃣ Define Your Tools

```python
def get_weather_forecast(location: str) -> dict:
    """Retrieves the weather forecast for a given location"""
    # ...your logic...
```

### 3️⃣ Load the Model

```python
import llama_cpp

model = llama_cpp.Llama(
    model_path="path/to/model.gguf",
    n_threads=4,  # Adjust for your CPU
    n_ctx=8192,
)
```

### 4️⃣ Compose Messages & Call Functions

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the temperature in a random city?"}
]
response = model.create_chat_completion(messages=messages)
print(response)
```

### 5️⃣ Parse and Execute Function Calls

- The model will return function calls in a structured format.
- Parse, execute, and feed results back for multi-step reasoning!

---

## 🗂️ Example Workflow

```
User Query
   |
   v
Llama Model (llama-cpp-python)
   |
   v
Is Function Call Needed?
   |         \
  Yes         No
   |           \
   v            v
Parse &     Final Answer
Execute
Function
   |
   v
Return Tool Result
   |
   v
(loop back to Llama Model)
```

---

## 🛠️ Example: Multi-Function Chaining

```python
# Model output:
# <function_calls>
# [
#   {"name": "get_random_city", "output": "random_city"},
#   {"name": "get_weather_forecast", "params": {"location": "random_city"}, "output": "temperature"}
# ]
# </function_calls>
```

---

## 💡 Tips

- **CPU Performance:** Increase `n_threads` for faster inference.
- **Context Length:** Adjust `n_ctx` for longer conversations.
- **Model Size:** Use smaller GGUF models for limited RAM.

---

## 🧑‍💻 Notebooks Included

- `examples/lllama-cpp-multiple-fn.ipynb` — Multi-function tool use
- `examples/ollama_openai_tools.ipynb` — Ollama API + OpenAI tools
- `examples/localai_api_fn_calling.ipynb` — LocalAI API demo

---

## ⚠️ Disclaimer

- This is **not** the official NousResearch repo.
- Most modifications were made by AI (GitHub Copilot) to enable CPU-only, llama-cpp-python-based workflows.
- For GPU or HuggingFace Transformers support, see the [original repo](https://github.com/NousResearch/Hermes-Function-Calling).

---

## 📝 License

MIT License (c) 2024 Nous Research

---

## 🤝 Acknowledgements

- [NousResearch](https://github.com/NousResearch) for the original Hermes-Function-Calling
- [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Ollama](https://ollama.com/)

---

## 🌟 Happy Function Calling! 🌟

