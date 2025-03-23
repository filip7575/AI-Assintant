"""
Punto de entrada principal para el asistente AI
"""
import os
from ai_assistant import AIAssistant

if __name__ == "__main__":
    # Establecer la variable de entorno para evitar advertencia de paralelismo
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Crear directorio para documentos si no existe
    docs_dir = "./documents"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Creado directorio para documentos: {docs_dir}")
    
    # Iniciar el asistente
    assistant = AIAssistant(docs_dir=docs_dir)
    assistant.run()
