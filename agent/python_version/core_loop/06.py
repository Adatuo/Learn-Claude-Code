import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Keep working step by step, and use compact if the conversation gets too long."
)

CONTEXT_LIMIT = 50000
KEEP_RECENT_TOOL_RESULTS = 3# 保留最近的tool_result块的数量
PERSIST_THRESHOLD = 3000 #持久阀值，塞入历史对话的最大值
PREVIEW_CHARS = 2000 #预览字符
# 将压缩存储到磁盘的隐藏目录下
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"\

@dataclass
class CompactState:
    has_compacted:bool = False
    last_summary:str = ""
    recent_files:list[str] = field(default_factory=list)

def estimate_context_size(messages:list) -> int:
    return len(str(messages))

# 跟踪最近读取的文件
def track_recent_file(state:CompactState,path:str) -> None:
    if path in state.recent_files:
        state.recent_files.remove(path)
    state.recent_files.append(path)
    if len(state.recent_files) > 5:
        state.recent_files[:] = state.recent_files[-5:]

def safe_path(path_str:str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path

# 持久化大型输出，并给出一个预览
def persist_large_output(tool_use_id:str,output:str) -> str:
    # 小于一定的字符直接返回给AI
    if len(output) <= PERSIST_THRESHOLD:
        return output
    # 这里是把工具执行结果存储到本地的一个文件中，只保留前一定数量的字符给AI看，大文件不直接给AI看,避免Token溢出
    TOOL_RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    stored_path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    # 检查文件是否已经存在，如果存在说明是tool_result重新执行，可以覆盖
    if not stored_path.exists():
        # 写到本地
        stored_path.write_text(output)
    
    preview = output[:PREVIEW_CHARS]
    # 相对路径，显示给ai
    rel_path = stored_path.relative_to(WORKDIR)
    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>"""
    )
    
# 收集tool_result块    
def collect_tool_result_blocks(messages:list) -> list[tuple[int,int,dict]]:
    blocks = []
    for messages_index,message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content,list):
            continue
        for block_index,block in enumerate(content):
            if isinstance(block,dict) and block.get("type") == "tool_result":
                blocks.append((messages_index,block_index,block))
    return blocks

# 微型压缩，只压缩tool_result，不压缩对话
def micro_compact(messages:list) -> list:
    tool_results = collect_tool_result_blocks(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages
    
    for _,_, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        content = block.get("content","")
        if not isinstance(content,str) or len(content) <= 120:
            continue
        # 对于更早的超过120字符的结果直接清空，并返回字符串
        block["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages

# 把完整的消息记录写成文件，这里是把用户的输入不做压缩直接存到本地
def write_transcript(messages:list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True,exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as handle:
        for message in messages:
            handle.write(json.dumps(message,default=str) + "\n")
    return path

def summarize_history(messages:list) -> str:
    # 把完整的消息记录发给一个独立的实例，让它总结对话。这个实例不处理工具调用，只负责阅读和摘要，能更好地把握上下文。为了防止摘要过长，截取了前 80k 字符。
    conversation = json.dumps(messages,default=str)[:80000]
    # 让AI去总结它
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve:\n"
        "1. The current goal\n"
        "2. Important findings and decisions\n"
        "3. Files read or changed\n"
        "4. Remaining work\n"
        "5. User constraints and preferences\n"
        "Be compact but concrete.\n\n"
        f"{conversation}"
    )
    # 独立的总结实例
    response = client.messages.create(model=MODEL,messages=[{"role":"user","content":prompt}],max_tokens=2000)
    return response.content[0].text.strip()

def compact_history(messages:list,state:CompactState,focus:str|None = None) -> list:
    transcript_path = write_transcript(messages)
    print(f"[transcript saved:{transcript_path}]")

    summary = summarize_history(messages)
    if focus:
        summary += f"\n\nFocus to preserve next: {focus}"
    if state.recent_files:
        recent_lines = "\n".join(f"- {path}" for path in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed:\n{recent_lines}"

    state.has_compacted = True
    state.last_summary = summary

    # 返回总结后的消息
    return [{
        "role": "user",
        "content": (
            "This conversation was compacted so the agent can continue working.\n\n"
            f"{summary}"
        ),
    }]

def run_bash(command: str, tool_use_id: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    output = (result.stdout + result.stderr).strip() or "(no output)"
    return persist_large_output(tool_use_id, output)
    
def run_read(path: str, tool_use_id: str, state: CompactState, limit: int | None = None) -> str:
    try:
        # 这里有个跟踪最近读取的文件
        track_recent_file(state, path)
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        output = "\n".join(lines)
        return persist_large_output(tool_use_id, output)
    except Exception as exc:
        return f"Error: {exc}"
def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "compact",
        "description": "Summarize earlier conversation so work can continue in a smaller context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {"type": "string"},
            },
        },
    },
]

def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()

# 函数执行工具，而不是字典，灵活注入不同的变量
def execute_tool(block, state: CompactState) -> str:
    if block.name == "bash":
        return run_bash(block.input["command"], block.id)
    if block.name == "read_file":
        return run_read(block.input["path"], block.id, state, block.input.get("limit"))
    if block.name == "write_file":
        return run_write(block.input["path"], block.input["content"])
    if block.name == "edit_file":
        return run_edit(block.input["path"], block.input["old_text"], block.input["new_text"])
    if block.name == "compact":
        return "Compacting conversation..."
    return f"Unknown tool: {block.name}"

def agent_loop(messages: list, state: CompactState) -> None:
    while True:
        # 微量压缩：长度大于120字符的工具结果被替换为摘要字符串
        messages[:] = micro_compact(messages)
        # 当长度过大的时候进行全量自动压缩
        if estimate_context_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages, state)
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        # 手动压缩
        manual_compact = False
        # 专注压缩
        compact_focus = None
        for block in response.content:
            if block.type != "tool_use":
                continue
            # 执行工具
            output = execute_tool(block, state)
            # AI判断是否需要手动压缩，以及专注的压缩点
            if block.name == "compact":
                manual_compact = True
                compact_focus = (block.input or {}).get("focus")
            print(f"> {block.name}: {str(output)[:200]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(output),
            })
        messages.append({"role": "user", "content": results})
        if manual_compact:
            print("[manual compact]")
            messages[:] = compact_history(messages, state, focus=compact_focus)

if __name__ == "__main__":
    history = []
    compact_state = CompactState()
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, compact_state)
        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()
    
    
        