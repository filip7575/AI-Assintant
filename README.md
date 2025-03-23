# AI-Assintant
AI Assistant with RAG and Code Execution
A command line AI assistant with RAG (Retrieval Augmented Generation) capabilities and secure execution of Python code and Bash commands.
Features

🤖 Ollama integration to access local LLM models

📚 RAG system to contextualize responses with local documents

🌐 Automatic web content indexing with the /ihtml command

🐍 Secure Python code execution with options to save or run

💻 Controlled Bash command execution with user confirmation

📊 Processing of different document types (PDF, Markdown, HTML)

🔄 Modular architecture for easy maintenance and extensibility

<img src="[ruta/a/la/imagen](https://github.com/filip7575/AI-Assintant/console.jpg" alt="Descripción alternativa" style="width:200px; height:auto;" />

# Project Structure

```
.
├── main.py                  # Entry point
├── ai_assistant.py          # Main assistant and interaction
├── rag_system.py            # RAG system and vector management
├── document_processor.py    # Document processing
├── utils.py                 # Shared utilities
├── documents/               # Directory for documents to index
└── chroma_db/               # Vector database (automatically generated)
```

# Requirements

Python 3.8+
Ollama installed and configured
Dependencies listed in requirements.txt

# Installation

### 1.- Clone the repository:
```
git clone https://github.com/filip7575/ai-assistant-rag.git
cd ai-assistant-rag
```

### 2.- Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3.- nstall dependencies:
```
pip install -r requirements.txt
```
### 4.- Make sure Ollama is installed and running

Visit ollama.ai for installation instructions
By default, the qwen2.5-coder:7b model is used, but you can change it



# Usage
Run the assistant:
```
python main.py
```

# Available Commands

- **Exit the assistant**: `/exit`
- **View available commands**: `/help`
- **Index web content**: `/ihtml URL`
- **Change Ollama model**: `/model NAME` (e.g.: `/model llama3`)
- **Re-index documents**: `/rag`

# Usage Examples

### 1.- Simple query:
```
What is a RAG system?
```

### 2.- Python code execution
```
Create code to ping google.com
```

### 3.- Bash command execution
```
Show me how to list all files in the current directory
```

### 4.- Index a webpage
```
/ihtml https://example.com
```

### RAG System
The assistant uses a RAG system to contextualize its responses with information from:

Documents in the ./documents directory: PDF, Markdown, text files
Indexed web pages: through the /ihtml URL command

When you make a query, the system finds the most relevant context in the indexed documents and provides it to the LLM model to generate more accurate and relevant responses.

# Extension and Customization
Thanks to the modular architecture, you can easily extend the assistant:

New document types: Extend document_processor.py
More commands: Add cases in the run() method of ai_assistant.py
Integration with other models: Modify the Ollama-related parts

# License
This project is available under the GNU General Public License v3.0 (GPL-3.0).
