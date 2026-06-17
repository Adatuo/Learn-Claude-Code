import json
import os
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 允许的权限
MODES = ("default","plan","auto")

# 只读工具
READ_ONLY_TOOLS = {"read_file","bash_readonly"}

# 询问权限
WRITE_TOOLS = {"write_file","bash","edit_file"}

class BashSecurityValidator:
    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),       # shell metacharacters
        ("sudo", r"\bsudo\b"),                 # privilege escalation
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),  # recursive -delete
        ("cmd_substitution", r"\$\("),          # command substitution
        ("ifs_injection", r"\bIFS\s*="),        # IFS manipulation
    ]

    # 检查命令中是否包含危险模式，并返回危险命令
    def validate(self,command:str) -> list:
        """Check command for dangerous patterns"""
        failures = []
        for name,pattern in self.VALIDATORS:
            if re.search(pattern, command):
                # 增加一个元组
                failures.append((name, pattern))
        return failures

    # 检查命令是否安全
    def is_safe(self,command:str) -> bool:
        # 返回命令中危险模式的数量，如果为0则安全
        return len(self.validate(command)) == 0
        
    def describe_failures(self,command:str) -> str:
        failures = self.validate(command)
        if not failures:
            return "No issues detected"
        parts = [f"{name} (pattern:{pattern})" for name,pattern in failures]
        return "Security flags: " + ", ".join(parts)

# 工作空间安全验
def is_workspace_trusted(workspace: Path = None) -> bool:
    ws = workspace or WORKDIR
    trust_markers = ws / ".claude" / ".claude_trusted"
    return trust_markers.exists()
        
# 实例化检测器
bash_validator = BashSecurityValidator()

#  工具默认使用的部分
DEFAULT_RULES = [
    # Always deny dangerous patterns
    {"tool": "bash", "content": "rm -rf /", "behavior": "deny"},
    {"tool": "bash", "content": "sudo *", "behavior": "deny"},
    # Allow reading anything
    {"tool": "read_file", "path": "*", "behavior": "allow"},
]

# 策略管理器
class PermissionManager:
    def __init__(self,mode:str = "default",rules:list = None):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from {MODES}")
        self.mode = mode
        self.rules = rules or list(DEFAULT_RULES)

        # 联机错误计数
        self.consecutive_denials = 0
        self.max_consecutive_denials = 3

    def check(self,tool_name:str,tool_input:dict) -> dict:
        if tool_name == "bash":
            command = tool_input.get("command","")
            failures = bash_validator.validate(command)
            if failures:
                # 检测高危命令
                severe = {"sudo","rm_rf"}
                severe_hits = [f for f in failures if f[0] in severe]
                if severe_hits:
                    # 检查是什么样的高危命令
                    desc = bash_validator.describe_failures(command)
                    return {"behavior":"ask","reason": f"Bash validator flagged: {desc}"}

        for rule in self.rules:
            if rule["behavior"] != "deny":
                continue
            if self._matches(rule,tool_name,tool_input):
                return {"behavior":"deny","reason": f"Blocked by deny rule:{rule}"}

        # agent 模式
        if self.mode == "plan":
            if tool_name in WRITE_TOOLS:
                return {"behavior":"deny","reason":"Plan mode:write operations are blocked"}
            return {"behavior": "allow", "reason": "Plan mode: read-only allowed"}

        if self.mode == "auto":
            if tool_name in READ_ONLY_TOOLS or tool_name == "read_file":
                return {"behavior": "allow","reason": "Auto mode: read-only tool auto-approved"}
            pass

        for rule in self.rules:
            if rule["behavior"] != "allow":
                continue
            if self._matches(rule,tool_name,tool_input):
                self.consecutive_denials = 0
                return {"behavior": "allow","reason":f"Mathed allow rule: {rule}"}

        # 找不到规则询问用户
        return {"behavior": "ask",
                "reason": f"No rule matched for {tool_name}, asking user"}

    # 询问用户这个回答哪一个更好
    def ask_user(self,tool_name: str,tool_input: dict) -> bool:
        # 转化成json
        preview = json.dumps(tool_input,ensure_ascii=False)[:200]
        print(f"\n  [Permission] {tool_name}: {preview}")
        try:
            answer = input("Allow? (y/n/always): ")
        except (EOFError,KeyboardInterrupt):
            return False

        if answer == 'always':
            self.rules.append({"tool":tool_name,"path":"*","behavior":"allow"})
            self.consecutive_denials = 0
            return True
        if answer in ("y","yes"):
            self.consecutive_denials = 0
            return True

        self.consecutive_denials += 1
        if self.consecutive_denials >= self.max_consecutive_denials:
            print(f"  [{self.consecutive_denials} consecutive denials -- " "consider switching to plan mode]")
        return False

    # 检查规则是否匹配
    def _matches(self,rule:dict,tool_name:str,tool_input:dict) -> bool:
        # 工具名称是否匹配
        if rule.get("tool") and rule["tool"] != "*":
            if rule["tool"] != tool_name:
                return False
        # 路径是否匹配
        if "path" in rule and rule["path"] != "*":
            path = tool_input.get("path","")
            if not fnmatch(path,rule["path"]):
                return False
        # 内容是否匹配
        if "content" in rule:
            command = tool_input.get("command","")
            if not fnmatch(command,rule["content"]):
                return False
        return True
                
# -- 常用工具 --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))# 找到第一个进行替换
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

# 工具注册
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]
SYSTEM = f"""You are a coding agent at {WORKDIR}. Use tools to solve tasks.
The user controls permissions. Some tool calls may be denied."""

def agent_loop(messages: list,perms:PermissionManager):
    while True:
        # 这几个必须要有才能调用SDK
        response = client.messages.create(model=MODEL,max_tokens=4096,system=SYSTEM,messages=messages,tools=TOOLS)
        messages.append({"role":"assistant","content":response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            decision = perms.check(block.name,block.input or {})

            if decision["behavior"] == "deny":
                output = f"[Permission denied] {decision['reason']}"
                print(f"  [DENIED] {block.name}: {decision['reason']}")
            
            elif decision["behavior"] == "ask":
                if perms.ask_user(block.name,block.input or {}):
                    handler = TOOL_HANDLERS[block.name]
                    # 传递参数调用工具
                    output = handler(**(block.input or {})) if handler else f"Unknown: {block.name}"
                    print(f"> {block.name}: {str(output)[:200]}")
                else:
                    output = f"Permission denied by user for {block.name}"
                    print(f"  [USER DENIED] {block.name}")
            # 允许运行
            else:
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**(block.input or {})) if handler else f"Unknown: {block.name}"
                print(f"> {block.name}: {str(output)[:200]}")
                
            results.append({
                "type":"tool_result",
                "tool_use_id":block.id,
                "content":str(output)
            })
        # 将所有结果发送给AI
        messages.append({"role":"user","content":results})

if __name__ == "__main__":
    print("Permission modes: default,plan,auto")
    mode_input = input("Mode (default): ").strip().lower() or "default"
    if mode_input not in MODES:
        mode_input = "default"
    perms = PermissionManager(mode_input)
    print(f"[Permission mode: {mode_input}]")

    history = []
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q","exit",""):
            break

        #  检测用户是否切换模式
        if query.startswith("/mode"):
            parts = query.split()
            if len(parts) == 2 and parts[1] in MODES:
                perms.mode = parts[1]
                print(f"[Switched to {parts[1]} mode]")
            else:
                print(f"Usage: /mode<{'|'.join(MODES)}>")
            continue
        
        #  检测是否对某条规则进行操作
        if query.strip() == "/rules":
            for i,rule in enumerate(perms.rules):
                print(f"{i}:{rule}")
            continue
                
        history.append({"role":"user","content":query})
        agent_loop(history,perms)
        response_content = history[-1]["content"]
        
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block,"text"):
                    print(block.text)
                    
        print()