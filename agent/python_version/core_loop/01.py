import os
import subprocess
# 这个库可以方便的生成数据结构的类
from dataclasses import dataclass
try:
    # 让input终端输入会支持方向键修改、上下翻历史记
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')#给命令行输入这个命令 off并且关掉CTRL+Z这些玩意
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv
# 加载配置文件,会自动去读取根目录下的.env
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command in the current workspace.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

@dataclass
class LoopState:
    # agent最小循环数据结构
    messages: list #不是聊天记录层，而是模型下一轮要读的工作上下文
    turn_count: int = 1# 这个是用来记录你已经执行了多少次循环
    transition_reason: str | None = None # 为什么要继续循环

# 执行bash命令
def run_bash(command: str) -> str:
    # 检查危险命令
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    # any有一个true，就返回true
    if any(item in command for item in dangerous):#command里面有没有dangerous
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"
    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(no output)"
    

def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""# 如果content不是list，就返回空字符串
    texts = []# 用来存储文本
    for block in content:
        text = getattr(block, "text", None)# 获取文本
        if text:
            texts.append(text)# 将文本添加到列表中
    return "\n".join(texts).strip()# 将列表中的文本拼接成一个字符串，并去除两端的空白字符

def execute_tool_calls(response_content) -> list[dict]:
    results = []
    for block in response_content:
        # 当前这个 block 不是 tool_use 时就跳过循环
        if block.type != "tool_use":
            continue
        command = block.input["command"]
        # 这行代码是在终端 / 命令行中打印黄色的命令文本，那些看起来像乱码的 \033[33m 和 \033[0m 是 ANSI 转义序列，专门用来控制终端的文本颜色、背景色和样式
        print(f"\033[33m$ {command}\033[0m")
        # 调用执行命令的函数
        output = run_bash(command)
        print(output[:200])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,# 这条结果对应的是你刚才哪一次工具调用
            "content": output,
        })
    return results

def run_one_turn(state: LoopState) -> bool:
    # 调用模型
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
    )
    # 给模型传递、说了什么
    state.messages.append({"role": "assistant", "content": response.content})
    # 如果模型没有调用工具，就返回False 退出循环
    if response.stop_reason != "tool_use":
        state.transition_reason = None
        return False
    results = execute_tool_calls(response.content)
    if not results:
        state.transition_reason = None
        return False
    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True

def agent_loop(state: LoopState) -> None:
    while run_one_turn(state):
        pass
# 既能当“可执行脚本”跑起来，又能被当模块复用（比如只复用函数）
if __name__ == "__main__":
    history = []
    while True:
        try:
            # 终端显示一个青色的 s01 >> 
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        # strip() 方法用于移除字符串两端的空白字符（包括空格、换行符、制表符等），并返回一个新的字符串。lower() 方法将字符串转换为小写。    
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)
        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()