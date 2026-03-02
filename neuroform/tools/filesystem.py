import os
import shutil
from pathlib import Path
from neuroform.tools.manager import tool_registry

# Safe bound to prevent reading enormous files or outputting too much
MAX_BYTES = 100 * 1024

def read_file(path: str) -> str:
    """Reads the contents of a file."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: File {path} does not exist."
        if not p.is_file():
            return f"Error: {path} is not a file."
            
        # Check size to prevent crashing LLM context
        if p.stat().st_size > MAX_BYTES:
            return f"Error: File is too large to read (> {MAX_BYTES} bytes)."
            
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(path: str, content: str) -> str:
    """Writes content to a file (overwrites existing)."""
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def append_to_file(path: str, content: str) -> str:
    """Appends content to the end of a file."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return write_file(path, content)
            
        with open(p, "a", encoding="utf-8") as f:
            f.write("\n" + content)
        return f"Success: Appended to {path}"
    except Exception as e:
        return f"Error appending to file: {str(e)}"

def list_directory(path: str) -> str:
    """Lists files and folders in a directory."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: Directory {path} does not exist."
        if not p.is_dir():
            return f"Error: {path} is not a directory."
            
        items = []
        for x in p.iterdir():
            suffix = "/" if x.is_dir() else ""
            items.append(f"{x.name}{suffix}")
            
        return "Contents:\n" + "\n".join(sorted(items))
    except Exception as e:
        return f"Error listing directory: {str(e)}"

# Register with ToolManager
tool_registry.register(
    func=read_file,
    description="Read the text content of a local file. Returns the content or an error string.",
    parameters={
        "path": {"type": "string", "description": "Absolute or relative path to the file"}
    }
)

tool_registry.register(
    func=write_file,
    description="Create or overwrite a file with text content. Automatically creates parent directories.",
    parameters={
        "path": {"type": "string", "description": "Absolute or relative path to the file"},
        "content": {"type": "string", "description": "The exact text content to write"}
    }
)

tool_registry.register(
    func=append_to_file,
    description="Append text to the end of an existing file.",
    parameters={
        "path": {"type": "string", "description": "Path to the file to append to"},
        "content": {"type": "string", "description": "Text to append"}
    }
)

tool_registry.register(
    func=list_directory,
    description="List the contents (files and folders) of a local directory.",
    parameters={
        "path": {"type": "string", "description": "Absolute or relative path to the directory"}
    }
)
