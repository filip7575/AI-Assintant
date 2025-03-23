"""
Módulo de utilidades compartidas para el asistente AI
"""
import io
import re
import subprocess
import contextlib
from typing import Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

console = Console()

def execute_bash_command(command: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Ejecuta comandos bash con confirmación de usuario.
    
    Args:
        command: Comando bash a ejecutar
        
    Returns:
        Tuple: (stdout, stderr) o (None, None) si se cancela
    """
    try:
        # Limpiar el comando de marcadores especiales y caracteres extraños
        clean_command = command.strip()
        
        # Eliminar cualquier prefijo $ que pueda haber quedado
        if clean_command.startswith('$'):
            clean_command = clean_command[1:].strip()
        
        console.print(Panel(f"Comando a ejecutar:\n{clean_command}", title="[yellow]Bash[/yellow]"))
        
        # Verificar comandos potencialmente peligrosos
        if any(cmd in clean_command.lower() for cmd in ['rm', 'mv', 'cp', 'chmod', 'sudo']):
            console.print("[red]⚠️  Advertencia: Este comando puede modificar archivos o el sistema.[/red]")
        
        if Confirm.ask("¿Desea ejecutar este comando?"):
            resultado = subprocess.run(
                clean_command,
                shell=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout_content = resultado.stdout or ""
            stderr_content = resultado.stderr or ""
            
            # Garantizar que la salida termina con una nueva línea
            if stdout_content and not stdout_content.endswith('\n'):
                stdout_content += '\n'
            if stderr_content and not stderr_content.endswith('\n'):
                stderr_content += '\n'
            
            if stdout_content:
                console.print(Panel(stdout_content, title="[green]Salida[/green]", expand=False))
            if stderr_content:
                console.print(Panel(stderr_content, title="[red]Errores[/red]", expand=False))
            
            return stdout_content, stderr_content
        else:
            console.print("[yellow]Ejecución cancelada[/yellow]")
            return None, None
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return None, str(e)

def execute_python_code(code: str) -> str:
    """
    Ejecuta código Python con confirmación de usuario.
    
    Args:
        code: Código Python a ejecutar
        
    Returns:
        str: Resultado de la ejecución o mensaje de error
    """
    output = io.StringIO()
    
    console.print(Panel(Syntax(code, "python", theme="monokai"), title="[yellow]Código Python[/yellow]"))
    
    response = Prompt.ask(
        "¿Qué deseas hacer con este código?",
        choices=["ejecutar", "guardar", "cancelar"],
        default="ejecutar"
    )
    
    if response == "cancelar":
        console.print("[yellow]Ejecución cancelada[/yellow]")
        return ""
        
    if response == "guardar":
        filename = Prompt.ask("Nombre del archivo para guardar", default="script.py")
        try:
            with open(filename, "w") as f:
                f.write(code)
            console.print(f"[green]Código guardado en {filename}[/green]")
            return f"Código guardado en {filename}"
        except Exception as e:
            error_msg = f"Error al guardar el archivo: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    # Si llegamos aquí, la opción es ejecutar
    try:
        with contextlib.redirect_stdout(output):
            exec(code, globals())
        result = output.getvalue()
        
        if result.strip():
            console.print(Panel(result, title="[green]Resultado[/green]"))
        else:
            console.print("[green]Código ejecutado sin salida[/green]")
        
        return result
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        return error_msg

def extract_code_blocks(text: str) -> Tuple[list, list]:
    """
    Extrae bloques de código bash y Python de un texto.
    
    Args:
        text: Texto del que extraer bloques de código
        
    Returns:
        Tuple: (lista de bloques bash, lista de bloques Python)
    """
    # Patrones para código bash
    bash_patterns = [
        r'```bash\n(.*?)```',                  # Estándar markdown
        r'```\n\$ (.*?)```',                   # Inicio con $ sin especificar bash
        r'```bash\n\$ (.*?)```',               # Inicio con $ especificando bash
    ]
    
    # Patrón para código Python
    python_patterns = [
        r'```python\n(.*?)```',                # Estándar markdown
        r'```py\n(.*?)```'                     # Variante abreviada
    ]
    
    # Extraer solo el primer bloque de cada tipo (para evitar múltiples ejecuciones)
    bash_blocks = []
    for pattern in bash_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            bash_blocks.append(matches[0])
            break
    
    python_blocks = []
    for pattern in python_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            python_blocks.append(matches[0])
            break
    
    # Limpiar y preparar bloques bash
    clean_bash_blocks = []
    for block in bash_blocks:
        # Obtener solo la primera línea de comando que no sea comentario
        command_lines = block.strip().split('\n')
        clean_command = None
        
        for line in command_lines:
            line = line.strip()
            # Ignorar líneas vacías o comentarios
            if not line or line.startswith('#'):
                continue
            # Eliminar prefijo $ si existe
            if line.startswith('$'):
                line = line[1:].strip()
            if line:
                clean_command = line
                break
                
        if clean_command:
            clean_bash_blocks.append(clean_command)
    
    # Limpiar bloques Python (reemplazar estilos de comillas si es necesario)
    clean_python_blocks = []
    for block in python_blocks:
        # Limpiar código Python
        code = block.strip()
        # Corregir __name__
        if "**name**" in code:
            code = code.replace("**name**", "__name__")
        clean_python_blocks.append(code)
    
    return clean_bash_blocks, clean_python_blocks
