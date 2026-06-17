import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

# 如果系统里已经有同名环境变量，不覆盖，保留原来的
load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    # 删 token 是为了在中转场景下避免 ANTHROPIC_AUTH_TOKEN 和 ANTHROPIC_API_KEY 冲突
    os.environ.pop("ANTHROPIC_AUTH_TOKEN",None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))# 自动去拿key
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use tools to solve tasks. Act, don't explain."
)

def safe_path(p:str) -> Path:
    path = (WORKDIR / p).resolve() # 将路径转换为绝对路径
    if not path.is_relative_to(WORKDIR):# 检查路径是否在工作目录下
        raise ValueError(f"Path escapes workspace: {p}")#抛出错误
    return path

def run_bash(command:str) -> str:
        # 检查危险命令
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    # any有一个true，就返回true
    if any(item in command for item in dangerous):#command里面有没有dangerous
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,# 把命令的输出抓进变量里，方便后续拼接、截断、返回给 LLM。
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip() #去掉空白字符
        # A if 条件 else B 这是python的语法，如果条件为真，则返回A，否则返回B
        return out[:50000] if out else "(no output)" # 截断到500字符
        # subprocess.TimeoutExpired时间超时的异常类
    except subprocess.TimeoutExpired:    
        return "Error: Timeout (120s)"

def run_read(path:str,limit:int=None) -> str:
    try:
        text = safe_path(path).read_text()# 直接把文件内容读成字符串
        lines = text.splitlines()# 把字符串按行分割成列表
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines)-limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path:str,content:str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True,exist_ok=True) # 递归创建文件所在目录，已有目录也不报错
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path:str,old_text:str,new_text:str)->str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text,new_text,1)) # 最后的 1 = 只替换 1 次（第一次匹配
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

# 安全的线程
CONCURRENCY_SAFE = {"read_file"}
# 不安全的线程
CONCURRENCY_UNSAFE = {"write_file","edit_file"}

TOOL_HANDLERS = {
    # lambda 匿名函数生成器 add = lambda x, y: x + y，x和y是参数，x+y是返回值
    # **kw 只是参数的一种特殊写法，表示将 kw 这个字典的所有键值对作为参数传递给函数。
    "bash": lambda **kw:run_bash(kw["command"]),
    "read_file": lambda **kw:run_read(kw["path"],kw.get("limit")),
    "write_file": lambda **kw:run_write(kw["path"],kw["content"]),
    "edit_file": lambda **kw:run_edit(kw["path"],kw["old_text"],kw["new_text"]),
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

# 这个函数的作用是：将消息列表转换为Anthropic API期望的格式。
def normalize_messages(messages:list) -> list:
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        # 格式分支处理
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            clean["content"] = [
                {k: v for k, v in block.items()
                 if not k.startswith("_")}
                for block in msg["content"]
                if isinstance(block, dict)
            ]
        else:
            clean["content"] = msg.get("content", "")
        cleaned.append(clean)

    existing_results = set()
    for msg in cleaned:
        if isinstance(msg.get("content"),list):
            for block in msg["content"]:
                if isinstance(block,dict) and block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))# 添加使用过的工具ID到集合中

    for msg in cleaned:
        if msg["role"] != "assistant" or not isinstance(msg.get('content'),list):# 如果消息不是来自于模型，或者内容不是列表，则跳过该消息
            continue
        for block in msg["content"]:
            if not isinstance(block,dict):# 如果块不是字典，则跳过该块
                continue
            if block.get("type") == "tool_use" and block.get("id") not in existing_results:# 如果块是工具使用，并且ID不在已使用过的工具ID集合中，则添加取消消息
                cleaned.append({"role":"user","content":[
                    {"type":"tool_result","tool_use_id":block["id"],"content":"(cancelled)"}
                ]})

    if not cleaned:
        return cleaned
    merged = [cleaned[0]]# 初始化合并后的消息列表，第一个消息作为初始值 
    for msg in cleaned[1:]:
        if msg["role"] == merged[-1]["role"]:# 如果消息的角色与合并后的最后一个消息的角色相同，则将消息合并到上一个消息中   
            prev = merged[-1]
            #这里的 \ 是行续行符
            prev_c = prev["content"] if isinstance(prev["content"],list) \
                else [{"type":"text","text":str(prev["content"])}]
            # 更常见的写法是用括号包住，但是这里用 \ 是为了避免语法错误
            curr_c = (
                msg["content"]
                if isinstance(msg["content"],list)
                else [{"type":"text","text":str(msg['content'])}]
            )
            prev["content"] = prev_c + curr_c
        else:
            merged.append(msg)
    return merged

def agent_loop(messages:list) -> list:
    while True:
        response = client.messages.create(model=MODEL,messages=normalize_messages(messages),tools=TOOLS,max_tokens=8000,system=SYSTEM)
        messages.append({"role":"assistant","content":response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool:{block.name}"
                print(f"> {block.name}:")
                print(output[:200])
                results.append({"type":"tool_result","tool_use_id":block.id,"content":output})
        messages.append({"role":"user","content":results})

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError,KeyboardInterrupt):
            break
        if query.strip().lower() in ("exit","q",""):
            break
        history.append({"role":"user","content":query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content,list):
            for block in response_content:
                # 用来判断：某个对象上有没有这个名字的属性，有就返回True，没有就返回False
                if hasattr(block,"text"):
                    print(block.text)
        print()
