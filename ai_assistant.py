"""
Módulo principal del asistente AI
"""
import os
from typing import List, Dict
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from utils import execute_bash_command, execute_python_code, extract_code_blocks
from rag_system import RAGSystem

console = Console()

class AIAssistant:
    """Clase principal del asistente de IA con capacidades de ejecución de código y RAG"""
    
    def __init__(self, docs_dir: str = "./documents", model: str = "qwen2.5-coder:7b"):
        """
        Inicializa el asistente de IA.
        
        Args:
            docs_dir: Directorio para documentos del sistema RAG
            model: Modelo de Ollama a utilizar
        """
        self.console = Console()
        self.running = True
        self.model = model
        self.docs_dir = docs_dir
        
        # Asegurar que existe el directorio de documentos
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            self.console.print(f"[yellow]Directorio '{self.docs_dir}' creado. Agrega tus documentos PDF o MD ahí.[/yellow]")
            
        # Inicializar el sistema RAG
        self.rag = RAGSystem()

    def interact_with_model(self, messages: List[Dict], query: str, processing_code: bool = False) -> List[Dict]:
        """
        Interactúa con el modelo LLM y procesa su respuesta.
        
        Args:
            messages: Historial de mensajes
            query: Consulta actual del usuario
            processing_code: Indica si ya estamos procesando código (para evitar recursión infinita)
            
        Returns:
            List[Dict]: Historial de mensajes actualizado
        """
        # Obtener contexto relevante del RAG solo si no estamos procesando código
        if not processing_code:
            # Verificar si el contexto está relacionado con los comandos del sistema
            system_commands = ['/help', '/exit', '/ihtml', '/model', '/rag', '/clear_rag']
            is_system_command = any(query.startswith(cmd) for cmd in system_commands)
            
            if not is_system_command:
                context = self.rag.get_relevant_context(query)
                
                # Hacer una copia del mensaje del sistema original para no modificarlo permanentemente
                current_messages = messages.copy()
                
                # Si hay contexto, añadirlo al mensaje del sistema para esta interacción
                if context:
                    enhanced_system_message = (
                        messages[0]["content"] + 
                        "\n\nContexto relevante para responder a la pregunta del usuario:\n" + 
                        context + 
                        "\n\nUtiliza este contexto para dar una respuesta precisa, pero no menciones explícitamente que estás usando contexto."
                    )
                    current_messages[0] = {"role": "system", "content": enhanced_system_message}
                    self.console.print("[dim]Se ha añadido contexto relevante a la consulta.[/dim]")
                else:
                    current_messages = messages.copy()
            else:
                # No necesitamos contexto para comandos del sistema
                current_messages = messages.copy()
        else:
            # Si estamos procesando código, usar los mensajes tal cual
            current_messages = messages.copy()

        history = current_messages
        response_complete = False
        
        while not response_complete:
            try:
                # Verificar que Ollama está disponible
                try:
                    stream = ollama.chat(
                        model=self.model,
                        messages=history,
                        stream=True
                    )
                except Exception as e:
                    self.console.print(f"[red]Error conectando con Ollama: {str(e)}[/red]")
                    self.console.print("[yellow]Asegúrate de que Ollama está instalado y el modelo está disponible.[/yellow]")
                    return messages + [{"role": "assistant", "content": f"Error conectando con Ollama: {str(e)}"}]
                
                full_response = ""
                self.console.print("\n[bold cyan]Respuesta:[/bold cyan]")
                
                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        response_part = chunk['message']['content']
                        full_response += response_part
                        self.console.print(response_part, end="")
                
                # Asegurar que hay un salto de línea después de la respuesta
                if full_response and not full_response.endswith('\n'):
                    self.console.print()
                
                # Añadir la respuesta completa al historial
                history.append({"role": "assistant", "content": full_response})
                
                # Solo procesar código si no estamos ya en una llamada recursiva
                if not processing_code:
                    # Verificar si hay código en la respuesta
                    try:
                        # Extraer bloques de código bash y Python
                        bash_blocks, python_blocks = extract_code_blocks(full_response)
                        
                        # Procesar solo un bloque de código (priorizar Python sobre bash)
                        code_processed = False
                        
                        # Primero intentar con Python
                        if python_blocks and not code_processed:
                            code = python_blocks[0]
                            result = execute_python_code(code)
                            if result:
                                execution_result = f"Resultado de la ejecución:\n{result}"
                                new_messages = messages.copy()
                                new_messages.append({"role": "assistant", "content": full_response})
                                new_messages.append({"role": "user", "content": execution_result})
                                code_processed = True
                                return self.interact_with_model(new_messages, execution_result, True)
                        
                        # Si no se procesó Python, intentar con bash
                        if bash_blocks and not code_processed:
                            command = bash_blocks[0]
                            output, error = execute_bash_command(command)
                            if output is not None or error is not None:
                                execution_result = f"Resultado de '{command}':\n{output or ''}{error or ''}"
                                new_messages = messages.copy()
                                new_messages.append({"role": "assistant", "content": full_response})
                                new_messages.append({"role": "user", "content": execution_result})
                                return self.interact_with_model(new_messages, execution_result, True)
                            
                    except Exception as e:
                        self.console.print(f"[red]Error procesando código: {str(e)}[/red]")
                
                response_complete = True

            except Exception as e:
                self.console.print(f"\n[red]Error en la interacción con el modelo: {e}[/red]")
                history.append({'role': 'assistant', 'content': f"Error: {str(e)}"})
                response_complete = True
                
        # Devolver el historial actualizado
        return history

    def run(self):
        """Ejecuta el asistente en modo interactivo"""
        self.console.print("[bold cyan]===== Asistente AI con RAG y ejecución de código =====\n[/bold cyan]")
        
        # Indexar documentos si hay alguno
        if os.path.exists(self.docs_dir) and any(os.path.isfile(os.path.join(self.docs_dir, f)) for f in os.listdir(self.docs_dir)):
            self.rag.index_documents(self.docs_dir)
        else:
            self.console.print(f"[yellow]No hay documentos para indexar en {self.docs_dir}[/yellow]")
            self.console.print(f"[yellow]Puedes agregar documentos PDF, Markdown o texto en ese directorio y usar /rag para indexarlos.[/yellow]")

        # Mensaje inicial del sistema
        system_message = """Eres un asistente AI con acceso a:

1. Python: ejecuta código Python con acceso al sistema.
   Ejemplo:
   ```python
   print("Hello World")
   ```

2. Bash: ejecuta comandos bash con acceso al sistema.
   Ejemplo:
   ```bash
   ls -la
   ```

3. Base de conocimientos: tienes acceso a una base de datos vectorial RAG (Retrieval Augmented Generation) que contiene documentos relevantes para proporcionar respuestas más precisas.

Cuando respondas preguntas basadas en el contexto proporcionado, integra la información naturalmente sin referirte explícitamente al contexto. Responde de manera útil, precisa y concisa a las consultas del usuario."""

        history = [{"role": "system", "content": system_message}]
        
        self.console.print(Panel(
            "/exit - Salir\n"
            "/help - Mostrar comandos disponibles\n"
            "/ihtml URL - Indexar contenido web\n"
            "/model NOMBRE - Cambiar modelo de Ollama (ej: /model llama3)\n"
            "/rag - Reindexar documentos\n"
            "/clear_rag - Limpiar base de datos RAG\n"
            "Escribe cualquier pregunta o solicitud de código",
            title="[yellow]Comandos disponibles[/yellow]"
        ))
        
        while self.running:
            try:
                user_input = Prompt.ask("\n[bold cyan]>>>[/bold cyan]")
                
                # Procesar comandos especiales
                if user_input.lower() == "/exit":
                    self.console.print("[yellow]¡Hasta pronto![/yellow]")
                    break
                    
                if user_input.lower() == "/help":
                    self.console.print(Panel(
                        "/exit - Salir\n"
                        "/help - Mostrar comandos disponibles\n"
                        "/ihtml URL - Indexar contenido web\n"
                        "/model NOMBRE - Cambiar modelo de Ollama (ej: /model llama3)\n"
                        "/rag - Reindexar documentos\n" 
                        "/clear_rag - Limpiar base de datos RAG\n"
                        "Escribe cualquier pregunta o solicitud de código",
                        title="[yellow]Comandos disponibles[/yellow]"
                    ))
                    continue
                
                # Procesar comando de indexación web
                if user_input.lower().startswith("/ihtml "):
                    url = user_input[7:].strip()  # Extraer la URL después de "/ihtml "
                    if url:
                        success = self.rag.add_html_from_url(url)
                        if success:
                            self.console.print(f"[green]La URL {url} ha sido indexada correctamente.[/green]")
                    else:
                        self.console.print("[red]Por favor, proporciona una URL válida después de /ihtml[/red]")
                    continue
                
                # Procesar comando de cambio de modelo
                if user_input.lower().startswith("/model "):
                    model_name = user_input[7:].strip()
                    if model_name:
                        self.model = model_name
                        self.console.print(f"[green]Modelo cambiado a {model_name}[/green]")
                    else:
                        self.console.print("[red]Por favor, proporciona un nombre de modelo válido[/red]")
                    continue
                
                # Procesar comando de reindexación
                if user_input.lower() == "/rag":
                    if os.path.exists(self.docs_dir):
                        success = self.rag.index_documents(self.docs_dir)
                        if success:
                            self.console.print("[green]Documentos indexados correctamente.[/green]")
                        else:
                            self.console.print("[yellow]No se pudieron indexar documentos. Verifica el directorio y los archivos.[/yellow]")
                    else:
                        self.console.print("[yellow]El directorio de documentos no existe.[/yellow]")
                    continue
                
                # Procesar comando para limpiar la base de datos RAG
                if user_input.lower() == "/clear_rag":
                    if Confirm.ask("[yellow]¿Estás seguro de que deseas limpiar toda la base de datos RAG?[/yellow]"):
                        success = self.rag.clear_database()
                        if success:
                            self.console.print("[green]Base de datos RAG limpiada correctamente.[/green]")
                    else:
                        self.console.print("[yellow]Operación cancelada.[/yellow]")
                    continue
                
                # Procesar consulta normal
                history.append({"role": "user", "content": user_input})
                history = self.interact_with_model(history, user_input)
                
            except KeyboardInterrupt:
                if Confirm.ask("\n¿Deseas salir?"):
                    self.console.print("[yellow]¡Hasta pronto![/yellow]")
                    break
                continue
            except Exception as e:
                self.console.print(f"[red]Error en el asistente: {str(e)}[/red]")
                # Mostrar traza completa para depuración
                import traceback
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
