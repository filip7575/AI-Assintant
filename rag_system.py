"""
Módulo para el sistema de recuperación aumentada de generación (RAG)
"""
import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console
from document_processor import DocumentProcessor

# Configurar variable de entorno para evitar advertencia de paralelismo
os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console()

class RAGSystem:
    """Sistema de recuperación aumentada de generación (RAG)"""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Inicializa el sistema RAG.
        
        Args:
            persist_dir: Directorio para persistir la base de datos vectorial
        """
        console.print("[yellow]Inicializando sistema RAG...[/yellow]")
        self.processor = DocumentProcessor()
        
        # Inicializar el modelo de embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            console.print("[green]Modelo de embeddings cargado correctamente.[/green]")
        except Exception as e:
            console.print(f"[red]Error al cargar el modelo de embeddings: {str(e)}[/red]")
            console.print("[yellow]Intentando usar configuración alternativa...[/yellow]")
            try:
                # Configuración alternativa con cache_folder
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    cache_folder="./embeddings_cache"
                )
                console.print("[green]Modelo de embeddings cargado con configuración alternativa.[/green]")
            except Exception as e2:
                console.print(f"[red]Error crítico al cargar embeddings: {str(e2)}[/red]")
                raise
        
        self.vector_store = None
        self.persist_directory = persist_dir
        
        # Crear el directorio de persistencia si no existe
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            console.print(f"[yellow]Creado directorio para la base de datos vectorial: {self.persist_directory}[/yellow]")
        
        # Inicializar vectorstore si existe
        if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                console.print("[green]Base de datos vectorial cargada correctamente.[/green]")
                # Verificar que hay documentos en la base de datos
                try:
                    count = len(self.vector_store._collection.get()['ids'])
                    console.print(f"[green]Base de datos contiene {count} documentos indexados.[/green]")
                except Exception as e:
                    console.print(f"[yellow]No se pudo determinar el número de documentos: {str(e)}[/yellow]")
            except Exception as e:
                console.print(f"[red]Error cargando base de datos vectorial: {str(e)}[/red]")
                console.print("[yellow]Creando una nueva base de datos vectorial.[/yellow]")
                self.vector_store = None
        else:
            console.print("[yellow]No se encontró base de datos vectorial existente. Se creará una nueva al indexar documentos.[/yellow]")

    def index_documents(self, document_dir: str) -> bool:
        """
        Indexa documentos en el directorio especificado.
        
        Args:
            document_dir: Directorio que contiene los documentos a indexar
            
        Returns:
            bool: True si la indexación fue exitosa, False en caso contrario
        """
        documents = []
        console.print("\n[bold yellow]Indexando documentos...[/bold yellow]")
        
        if not os.path.exists(document_dir):
            console.print(f"[red]El directorio {document_dir} no existe.[/red]")
            return False
            
        files_found = False
        
        # Listar todos los archivos en el directorio
        all_files = os.listdir(document_dir)
        if not all_files:
            console.print(f"[yellow]El directorio {document_dir} está vacío.[/yellow]")
            return False
            
        console.print(f"[dim]Encontrados {len(all_files)} archivos en el directorio.[/dim]")
        
        for filename in all_files:
            filepath = os.path.join(document_dir, filename)
            if not os.path.isfile(filepath):
                continue
                
            files_found = True
            
            try:
                if filename.lower().endswith('.pdf'):
                    docs = self.processor.load_pdf(filepath)
                    if docs:
                        documents.extend(docs)
                        console.print(f"[green]PDF procesado: {filename} - {len(docs)} fragmentos[/green]")
                elif filename.lower().endswith(('.md', '.txt')):
                    docs = self.processor.load_markdown(filepath)
                    if docs:
                        documents.extend(docs)
                        console.print(f"[green]Texto procesado: {filename} - {len(docs)} fragmentos[/green]")
            except Exception as e:
                console.print(f"[red]Error procesando {filename}: {str(e)}[/red]")
        
        if not files_found:
            console.print("[yellow]No se encontraron archivos para indexar en el directorio.[/yellow]")
            return False

        if not documents:
            console.print("[yellow]No se pudo extraer contenido de los documentos.[/yellow]")
            return False
            
        console.print(f"[bold green]Se extrajeron {len(documents)} fragmentos de texto de los documentos.[/bold green]")
        
        try:
            # Si ya existe un vectorstore, añadir documentos
            if self.vector_store:
                console.print("[yellow]Añadiendo documentos a la base de datos vectorial existente...[/yellow]")
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
                console.print("[green]Documentos añadidos a la base de datos existente.[/green]")
            else:
                console.print("[yellow]Creando nueva base de datos vectorial...[/yellow]")
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                console.print("[green]Nueva base de datos vectorial creada.[/green]")
                
            # Verificar que la indexación fue exitosa
            try:
                count = len(self.vector_store._collection.get()['ids'])
                console.print(f"[bold green]¡Indexación completada! Base de datos contiene ahora {count} documentos.[/bold green]")
            except Exception as e:
                console.print(f"[yellow]La indexación parece exitosa pero no se pudo verificar el conteo: {str(e)}[/yellow]")
                
            return True
        except Exception as e:
            console.print(f"[red]Error en la indexación: {str(e)}[/red]")
            return False

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Obtiene contexto relevante desde la base de datos vectorial para una consulta.
        
        Args:
            query: Consulta para la que se busca contexto
            k: Número de documentos a recuperar
            
        Returns:
            str: Contexto relevante como texto
        """
        if not self.vector_store:
            console.print("[yellow]No hay documentos indexados para buscar contexto.[/yellow]")
            return ""
        
        try:
            console.print(f"[dim]Buscando contexto relevante para: '{query}'[/dim]")
            
            # Usar búsqueda de similitud con score
            search_results = self.vector_store.similarity_search_with_score(query, k=k)
            
            if not search_results:
                console.print("[yellow]No se encontró contexto relevante.[/yellow]")
                return ""
                
            # Construir el contexto con información del score
            context_parts = []
            for idx, (doc, score) in enumerate(search_results):
                # Normalizar el score (dependiendo del tipo, puede ser distancia o similitud)
                relevance = round((1 - score) * 100) if score > 1 else round(score * 100)
                source = doc.metadata.get('source', 'desconocida')
                
                # Añadir encabezado con información sobre la relevancia
                header = f"[Fragmento {idx+1} | Relevancia: {relevance}% | Fuente: {source}]"
                context_parts.append(f"{header}\n{doc.page_content}")
            
            context = "\n\n---\n\n".join(context_parts)
            console.print(f"[green]Encontrados {len(search_results)} fragmentos relevantes para la consulta.[/green]")
            
            # Imprimir una breve descripción de lo que se encontró
            for i, (doc, _) in enumerate(search_results[:2]):  # Solo los primeros 2 para no sobrecargar
                source = doc.metadata.get('source', 'desconocida')
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                console.print(f"[dim]Resultado {i+1}: {source} - '{preview}'[/dim]")
            
            return context
        except Exception as e:
            console.print(f"[red]Error recuperando contexto: {str(e)}[/red]")
            # Imprimir más detalles para diagnóstico
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return ""
        
    def add_html_from_url(self, url: str) -> bool:
        """
        Añade contenido HTML desde una URL al vector store.
        
        Args:
            url: URL del contenido HTML a añadir
            
        Returns:
            bool: True si la indexación fue exitosa, False en caso contrario
        """
        try:
            console.print(f"[yellow]Procesando HTML de {url}...[/yellow]")
            documents = self.processor.load_html_from_url(url)
            
            if not documents:
                console.print("[red]No se pudo extraer contenido del HTML.[/red]")
                return False
                
            console.print(f"[green]Se extrajeron {len(documents)} fragmentos de texto de la URL.[/green]")
            
            try:
                if self.vector_store is None:
                    console.print("[yellow]Creando nueva base de datos vectorial para el contenido HTML...[/yellow]")
                    self.vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    console.print("[green]Nueva base de datos vectorial creada.[/green]")
                else:
                    console.print("[yellow]Añadiendo contenido HTML a la base de datos existente...[/yellow]")
                    self.vector_store.add_documents(documents)
                    self.vector_store.persist()
                    console.print("[green]Contenido HTML añadido a la base de datos.[/green]")
                
                # Verificar la indexación
                try:
                    count = len(self.vector_store._collection.get()['ids'])
                    console.print(f"[bold green]HTML de {url} indexado exitosamente! Base de datos contiene ahora {count} documentos.[/bold green]")
                except Exception as e:
                    console.print(f"[yellow]La indexación parece exitosa pero no se pudo verificar el conteo: {str(e)}[/yellow]")
                    
                return True
            except Exception as e:
                console.print(f"[red]Error al guardar en la base de datos vectorial: {str(e)}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error indexando HTML: {str(e)}[/red]")
            return False
            
    def clear_database(self) -> bool:
        """
        Limpia la base de datos vectorial.
        
        Returns:
            bool: True si la limpieza fue exitosa, False en caso contrario
        """
        try:
            if self.vector_store:
                console.print("[yellow]Limpiando base de datos vectorial...[/yellow]")
                # Recrear la colección (forma más segura de limpiar)
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="langchain",
                    collection_metadata={"hnsw:space": "cosine"}
                )
                self.vector_store.persist()
                console.print("[green]Base de datos vectorial limpiada correctamente.[/green]")
                return True
            else:
                console.print("[yellow]No hay base de datos vectorial para limpiar.[/yellow]")
                return False
        except Exception as e:
            console.print(f"[red]Error al limpiar la base de datos: {str(e)}[/red]")
            return False
