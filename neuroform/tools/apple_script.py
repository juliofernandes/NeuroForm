import subprocess
from neuroform.tools.manager import tool_registry

def _osascript(script: str) -> str:
    """Executes pure AppleScript code via osascript."""
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip() or "Success"
    except Exception as e:
        return f"Script execution failed: {str(e)}"

def create_apple_note(title: str, content: str) -> str:
    """Creates a new note in Apple Notes."""
    script = f'''
    tell application "Notes"
        tell account "iCloud"
            make new note at default folder with properties {{name:"{title}", body:"<h1>{title}</h1><p>{content}</p>"}}
        end tell
    end tell
    '''
    # Fallback to local 'On My Mac' if iCloud is not the top account
    script_fallback = f'''
    tell application "Notes"
        make new note at folder "Notes" with properties {{name:"{title}", body:"<h1>{title}</h1><p>{content}</p>"}}
    end tell
    '''
    res = _osascript(script)
    if "Error" in res:
        return _osascript(script_fallback)
    return res

def create_apple_reminder(list_name: str, task: str, body: str = "") -> str:
    """Creates a reminder in Apple Reminders."""
    body_prop = f', body:"{body}"' if body else ""
    script = f'''
    tell application "Reminders"
        tell list "{list_name}"
            make new reminder with properties {{name:"{task}"{body_prop}}}
        end tell
    end tell
    '''
    return _osascript(script)

def send_imessage(target: str, message: str) -> str:
    """Sends an iMessage. Target can be phone number or email."""
    script = f'''
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        set targetBuddy to buddy "{target}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    return _osascript(script)

# Register tools
tool_registry.register(
    func=create_apple_note,
    description="Create a native macOS Apple Note.",
    parameters={
        "title": {"type": "string", "description": "Title/Heading of the note"},
        "content": {"type": "string", "description": "HTML or plain text body of the note"}
    }
)

tool_registry.register(
    func=create_apple_reminder,
    description="Create a native macOS Apple Reminder.",
    parameters={
        "list_name": {"type": "string", "description": "Name of the Reminders list (e.g. 'Reminders' or 'Groceries')"},
        "task": {"type": "string", "description": "The task name/title"},
        "body": {"type": "string", "description": "Optional extended description"}
    }
)

tool_registry.register(
    func=send_imessage,
    description="Send a message natively through macOS Messages app (iMessage).",
    parameters={
        "target": {"type": "string", "description": "Phone number or email address of the recipient"},
        "message": {"type": "string", "description": "The exact message to send"}
    }
)
