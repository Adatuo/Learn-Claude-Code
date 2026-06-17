import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN",None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
PLAN_REMINDER_INTERVAL = 3 #3轮对话提醒一次

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool for multi-step work.
Keep exactly one step in_progress when a task has multiple steps.
Refresh the plan as work advances. Prefer tools over prose."""

@dataclass
class PlanItem:
    content:str
    status:str="pending"
    active_form:str=""

@dataclass
class PlanningState:
    items:list[PlanItem]=field(default_factory=list)# 给items赋一个默认值，默认值是一个空列表
    rounds_since_update: int=0# 给rounds_since_update赋一个默认值，默认值是0

# 任务管理器,负责管理任务的计划和进度.最重要的是防止会话飘逸
class TodoManager:
    def __init__(self):
        self.state = PlanningState()
    
    def update(self,items:list) -> str:
        if len(items) > 12:
            raise ValueError("Keep the session plan short(max 12 items)")
        
        normalized = [] # 规范化消息存储
        in_progress_count = 0 # 正在进行的任务数量
        for index,raw_item in enumerate(items): #在遍历列表（或其它可迭代对象）时，同时拿到下标和元素
            content = str(raw_item.get("content","")).strip()
            status = str(raw_item.get("status","pending")).strip()
            active_form = str(raw_item.get("activeForm","")).strip()

            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in ("pending","in_progress","completed"):
                raise ValueError(f"Item {index}: invalid status {status}")
            if status == "in_progress":
                in_progress_count += 1

            # 都在的话就格式化消息
            normalized.append(PlanItem(content=content,status=status,active_form=active_form))

        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")

        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str|None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>" # <reminder> 在这里是 Agent 用的提示字符串约定，不是前端标签

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet."
        
        lines = []

        for item in self.state.items:
            marker = {
                "pending":"[ ]",
                "in_progress":"[>]",
                "completed":"[x]"
            }[item.status] # 「字典字面量 + 键下标访问」
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)

        # 生成器表达式自带两个括号，但是外层的调用函数只有一个参数，所以可以省略一个括号
        completed = sum(1 for item in self.state.items if item.status == "completed")# 每满足一次“for 和 if”的条件，就加上一个 1，最终计算出满足条件的总数
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)

TODO = TodoManager()

def safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path
def run_bash(command: str) -> str:
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
    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(no output)"
def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
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
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
}

def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""# 如果content不是list，就返回空字符串
    texts = []# 用来存储文本
    for block in content:
        text = getattr(block, "text", None)# 获取文本
        if text:
            texts.append(text)# 将文本添加到列表中
    return "\n".join(texts).strip()# 将列表中的文本拼接成一个字符串，并去除两端的空白字符

TOOLS = [
    {
        "name":"bash",
        "description":"Run a shell command.",
        "input_schema":{
            "type":"object",
            "properties":{"command":{"type":"string"}},
            "required":["command"]
        }
    },
    {
        "name":"read_file",
        "description":"Read file contents.",
        "input_schema":{
            "type":"object",
            "properties":{
                "path":{"type":"string"},
                "limit":{"type":"integer"}
            },
            "required":["path"],
        }
    },
    {
        "name":"write_file",
        "description":"Write to a file.",
        "input_schema":{
            "type":"object",
            "properties":{
                "path":{"type":"string"},
                "content":{"type":"string"}
            },
            "required":["path","content"]
        }
    },
    {
        "name":"edit_file",
        "description":"Replace exact text in a file once.",
        "input_schema":{
            "type":"object",
            "properties":{
                "path":{"type":"string"},
                "old_text":{"type":"string"},
                "new_text":{"type":"string"}
            },
            "required":["path","old_text","new_text"]
        }
    },
    {
        "name":"todo",
        "description":"Rewrite the current plan for multi-step work.",
        "input_schema":{
            "type":"object",
            "properties":{
                "items":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "content":{"type":"string"},
                            "status":{"type":"string","enum":["pending","in_progress","completed"]},
                            "activeForm":{"type":"string","description":"Oprional present-continuous label."}
                        },
                        "required":["content","status"]
                    }
                }
            },
            "required":["items"]
        }
    }
]

#  Client SDK + 手写 agent loop
def agent_loop(messages:list) -> None:
    while True:
        response = client.messages.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
            system=SYSTEM,
        )
        messages.append({"role":"assistant","content":response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        used_todo = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = TOOL_HANDLERS.get(block.name)
            try:
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
            except Exception as exc:
                output = f"Error: {exc}"

            print(f"> {block.name}: {str(output)[:200]}") # 只取前 200 个字符,不足200的全部显示
            results.append({
                "type":"tool_result",
                "tool_use_id":block.id,
                "content":str(output),
            })
            if block.name == "todo":
                used_todo = True

        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            # 记录没有更新计划的时间，用于提醒用户更新计划
            TODO.note_round_without_update()
            # 提醒用户更新计划
            reminder = TODO.reminder()
            if reminder:
                results.insert(0,{"type":"text","content":reminder})

        messages.append({"role":"user","content":results})

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError,KeyboardInterrupt):
            break
        if query.strip().lower() in ("exit","q",""):
            break
        history.append({"role":"user","content":query})
        agent_loop(history)

        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()