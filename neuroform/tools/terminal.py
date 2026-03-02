import subprocess
from neuroform.tools.manager import tool_registry

def run_shell_command(command: str) -> str:
    """Runs a shell command on the host securely and returns stdout/stderr."""
    # Basic guard rails to prevent obvious system destruction
    dangerous = ["rm -rf /", "mkfs", "> /dev/nvme", "> /dev/sda"]
    if any(d in command for d in dangerous):
        return "Error: Command flagged as highly destructive and blocked."
        
    try:
        # Run with timeout to prevent hanging bots
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + "\n" + result.stderr
        output = output.strip()
        
        if not output:
            return "[Command succeeded with no output]"
            
        if len(output) > 10000:
            return output[:10000] + "\n...[Output truncated]"
            
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

# Register tools
tool_registry.register(
    func=run_shell_command,
    description="Execute a shell command on the host machine (macOS/zsh environment). Can be used to run Python scripts, git commands, tree, etc.",
    parameters={
        "command": {"type": "string", "description": "The terminal command to execute (e.g. 'ls -la' or 'python script.py')"}
    }
)
