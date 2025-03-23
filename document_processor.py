"""
Módulo para el procesamiento de documentos (PDF, Markdown, HTML)
"""
import os
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rich.console import Console

console = Console()

class DocumentProcessor:
    """Clase encargada del procesamiento de diferentes tipos de documentos"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa el procesador de documentos con parámetros de fragmentación.
        
        Args:
            chunk_size: Tamaño de los fragmentos de texto
            chunk_overlap: Superposición entre fragmentos
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
    
    def load_pdf(self, pdf_path: str):
        """
        Carga y procesa un documento PDF.
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Lista de documentos fragmentados
        """
        try:
            console.print(f"Procesando PDF: {os.path.basename(pdf_path)}")
            
            # Verificar que el archivo existe y es accesible
            if not os.path.exists(pdf_path):
                console.print(f"[red]El archivo PDF {pdf_path} no existe.[/red]")
                return []
                
            # Verificar que el archivo es realmente un PDF
            try:
                with open(pdf_path, 'rb') as f:
                    header = f.read(5)
                    if header != b'%PDF-':
                        console.print(f"[red]El archivo {pdf_path} no parece ser un PDF válido.[/red]")
                        return []
            except Exception as e:
                console.print(f"[red]Error al verificar el PDF {pdf_path}: {str(e)}[/red]")
                return []
            
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            result = self.text_splitter.split_documents(pages)
            console.print(f"[green]PDF procesado: {len(result)} fragmentos extraídos[/green]")
            return result
        except Exception as e:
            console.print(f"[red]Error procesando PDF {pdf_path}: {str(e)}[/red]")
            return []
    
    def load_markdown(self, md_path: str):
        """
        Carga y procesa un documento Markdown o texto plano.
        
        Args:
            md_path: Ruta al archivo Markdown o texto
            
        Returns:
            Lista de documentos fragmentados
        """
        try:
            console.print(f"Procesando texto: {os.path.basename(md_path)}")
            
            # Verificar que el archivo existe y es accesible
            if not os.path.exists(md_path):
                console.print(f"[red]El archivo {md_path} no existe.[/red]")
                return []
                
            # Verificar que el archivo es legible
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read(100)  # Leer solo los primeros 100 caracteres para verificar
            except Exception as e:
                console.print(f"[red]Error al leer el archivo {md_path}: {str(e)}[/red]")
                return []
            
            loader = TextLoader(md_path, encoding='utf-8')
            documents = loader.load()
            result = self.text_splitter.split_documents(documents)
            console.print(f"[green]Texto procesado: {len(result)} fragmentos extraídos[/green]")
            return result
        except Exception as e:
            console.print(f"[red]Error procesando texto {md_path}: {str(e)}[/red]")
            return []
    
    def load_html_from_url(self, url: str):
        """
        Carga y procesa contenido HTML desde una URL.
        
        Args:
            url: URL del contenido HTML a procesar
            
        Returns:
            Lista de documentos fragmentados
        """
        try:
            # Validar URL
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError("Formato de URL inválido")

            console.print(f"[yellow]Descargando contenido de {url}...[/yellow]")
            
            # Hacer petición HTTP
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parsear HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Eliminar elementos no deseados
            for element in soup(['script', 'style', 'meta', 'noscript', 'header', 'footer', 'nav']):
                element.decompose()

            # Extraer texto relevante
            text_content = []
            
            # Procesar título
            if soup.title:
                text_content.append(f"Title: {soup.title.string}")

            # Procesar meta descripción
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc and 'content' in meta_desc.attrs:
                text_content.append(f"Description: {meta_desc['content']}")

            # Procesar encabezados y párrafos
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
                if tag.text.strip():
                    text_content.append(tag.text.strip())

            # Unir texto
            full_text = f"URL: {url}\n\n" + "\n".join(text_content)
            
            # Dividir en fragmentos
            text_chunks = self.text_splitter.split_text(full_text)
            
            # Verificar que hay contenido
            if not text_chunks:
                console.print(f"[yellow]Advertencia: No se extrajo texto de {url}[/yellow]")
                return []
                
            # Convertir a formato documento
            documents = [Document(page_content=chunk, metadata={"source": url}) for chunk in text_chunks]
            
            console.print(f"[green]URL procesada: {len(documents)} fragmentos extraídos[/green]")
            return documents

        except Exception as e:
            console.print(f"[red]Error procesando HTML de {url}: {str(e)}[/red]")
            return []
