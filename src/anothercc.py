#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnotherCC - бесплатный ИИ-помощник для программирования в терминале.
Агент с инструментами (чтение/запись/правка файлов, поиск, команды, веб),
работает с любым OpenAI-совместимым API: OpenRouter (бесплатные модели),
Groq, Gemini, Cerebras, Ollama (локально) и др.
Только стандартная библиотека Python 3.8+.
"""
import json
import os
import re
import subprocess
import sys
import time
import fnmatch
import platform
import threading
import urllib.request
import urllib.error
import urllib.parse

VERSION = "1.0"
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".anothercc")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

PROVIDERS = [
    ("OpenRouter (бесплатные модели)", "https://openrouter.ai/api/v1", "openrouter/free",
     "Ключ: https://openrouter.ai/keys (бесплатно)"),
    ("Groq (бесплатный тариф)", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b",
     "Ключ: https://console.groq.com/keys"),
    ("Google Gemini (бесплатный тариф)", "https://generativelanguage.googleapis.com/v1beta/openai",
     "gemini-2.5-flash", "Ключ: https://aistudio.google.com/apikey"),
    ("Cerebras (бесплатный тариф)", "https://api.cerebras.ai/v1", "gpt-oss-120b",
     "Ключ: https://cloud.cerebras.ai"),
    ("Ollama (локально, без ключа)", "http://localhost:11434/v1", "qwen2.5-coder:7b",
     "Нужен установленный Ollama: https://ollama.com"),
    ("Свой OpenAI-совместимый API", "", "", "Укажете адрес, ключ и модель сами"),
]

MAX_TOOL_OUTPUT = 20000
MAX_HISTORY_CHARS = 240000
IS_WIN = os.name == "nt"

# ---------------------------------------------------------------- console --

def _enable_console():
    if IS_WIN:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleOutputCP(65001)
            k.SetConsoleCP(65001)
            h = k.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)
        except Exception:
            pass
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text, code):
    return "\033[%sm%s\033[0m" % (code, text) if USE_COLOR else text


def dim(t): return c(t, "2")
def bold(t): return c(t, "1")
def red(t): return c(t, "31")
def green(t): return c(t, "32")
def yellow(t): return c(t, "33")
def cyan(t): return c(t, "36")
def magenta(t): return c(t, "35")


def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        raise SystemExit(0)

# ----------------------------------------------------------------- config --

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def setup_wizard(cfg):
    print(bold("\nНастройка AnotherCC"))
    print("Выберите провайдера ИИ:\n")
    for i, p in enumerate(PROVIDERS, 1):
        print("  %d) %s" % (i, p[0]))
    while True:
        s = ask("\nНомер [1]: ").strip() or "1"
        if s.isdigit() and 1 <= int(s) <= len(PROVIDERS):
            break
    name, base, model, hint = PROVIDERS[int(s) - 1]
    print(dim(hint))
    if not base:
        base = ask("Адрес API (например https://host/v1): ").strip().rstrip("/")
    key = ""
    if "localhost" not in base and "127.0.0.1" not in base:
        key = ask("API-ключ: ").strip()
    else:
        key = ask("API-ключ (Enter, если не нужен): ").strip()
    m = ask("Модель [%s]: " % (model or "обязательно") ).strip() or model
    cfg.update({"base_url": base.rstrip("/"), "api_key": key, "model": m})
    save_config(cfg)
    print(green("Сохранено: " + CONFIG_PATH))
    return cfg

# -------------------------------------------------------------- HTTP / LLM --

class APIError(Exception):
    def __init__(self, status, body):
        Exception.__init__(self, "HTTP %s: %s" % (status, body[:500]))
        self.status = status
        self.body = body


class Cancelled(Exception):
    pass


def http_json(url, payload=None, headers=None, timeout=300, method=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "AnotherCC/" + VERSION)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        err = APIError(e.code, body)
        err.retry_after = e.headers.get("Retry-After") if e.headers else None
        raise err
    except urllib.error.URLError as e:
        raise APIError(0, "нет связи с сервером: %s" % e.reason)


def call_cancellable(fn, *args, **kwargs):
    box = {}

    def runner():
        try:
            box["r"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa
            box["e"] = e
    th = threading.Thread(target=runner, daemon=True)
    th.start()
    while th.is_alive():
        th.join(0.2)
        if CANCEL.is_set():
            raise Cancelled()
    if "e" in box:
        raise box["e"]
    return box["r"]


class LLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.native_tools = cfg.get("native_tools", True)

    def headers(self):
        h = {}
        if self.cfg.get("api_key"):
            h["Authorization"] = "Bearer " + self.cfg["api_key"]
        if "openrouter" in self.cfg.get("base_url", ""):
            h["X-Title"] = "AnotherCC"
            h["HTTP-Referer"] = "https://github.com/anothercc"
        return h

    def chat(self, messages, tools=None):
        payload = {"model": self.cfg["model"], "messages": messages, "temperature": 0.2}
        if tools and self.native_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = self.cfg["base_url"] + "/chat/completions"
        delay = 3
        for attempt in range(6):
            try:
                res = call_cancellable(http_json, url, payload, self.headers())
                if "choices" not in res or not res["choices"]:
                    err = res.get("error") or res
                    raise APIError(200, json.dumps(err, ensure_ascii=False))
                return res
            except APIError as e:
                low = e.body.lower()
                if tools and self.native_tools and e.status in (400, 404, 422) and (
                        "tool" in low or "function" in low):
                    print("\r" + " " * 20 + "\r" + dim("  модель не поддерживает вызов функций - перехожу на текстовый режим"))
                    self.native_tools = False
                    payload.pop("tools", None)
                    payload.pop("tool_choice", None)
                    return self.chat(messages, tools)
                if e.status in (0, 408, 429, 500, 502, 503, 504) and attempt < 5:
                    wait = delay
                    ra = getattr(e, "retry_after", None)
                    if ra and str(ra).isdigit():
                        wait = min(int(ra), 60)
                    print(dim("  сервер занят (%s), повтор через %dс..." % (e.status, wait)))
                    for _ in range(wait * 2):
                        if CANCEL.is_set():
                            raise Cancelled()
                        time.sleep(0.5)
                    delay = min(delay * 2, 40)
                    continue
                raise

    def list_models(self):
        res = http_json(self.cfg["base_url"] + "/models", headers=self.headers(), timeout=30)
        out = []
        for m in res.get("data", []):
            mid = m.get("id") or m.get("name")
            free = False
            pr = m.get("pricing")
            if isinstance(pr, dict):
                try:
                    free = float(pr.get("prompt", 1)) == 0 and float(pr.get("completion", 1)) == 0
                except Exception:
                    pass
            out.append((mid, free))
        return out

# ------------------------------------------------------------------ tools --

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vs"}


def trunc(s, n=MAX_TOOL_OUTPUT):
    if len(s) <= n:
        return s
    h = n * 2 // 3
    return s[:h] + "\n...[обрезано %d символов]...\n" % (len(s) - n) + s[-(n - h):]


def _p(path):
    return os.path.abspath(os.path.expanduser(path))


def t_read_file(path, offset=1, limit=2000):
    p = _p(path)
    if not os.path.isfile(p):
        return "ОШИБКА: файл не найден: " + p
    with open(p, "rb") as f:
        raw = f.read()
    if b"\0" in raw[:4096]:
        return "ОШИБКА: бинарный файл (%d байт)" % len(raw)
    lines = raw.decode("utf-8", "replace").splitlines()
    offset = max(1, int(offset or 1))
    limit = max(1, int(limit or 2000))
    chunk = lines[offset - 1: offset - 1 + limit]
    out = "\n".join("%6d\t%s" % (offset + i, l) for i, l in enumerate(chunk))
    if offset - 1 + limit < len(lines):
        out += "\n...[ещё %d строк, используйте offset]" % (len(lines) - offset + 1 - limit)
    return out or "(пустой файл)"


def t_write_file(path, content):
    p = _p(path)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return "OK: записано %d байт в %s" % (len(content.encode("utf-8")), p)


def t_edit_file(path, old_string, new_string, replace_all=False):
    p = _p(path)
    if not os.path.isfile(p):
        return "ОШИБКА: файл не найден: " + p
    with open(p, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    crlf = "\r\n" in text
    if crlf:
        text_n = text.replace("\r\n", "\n")
    else:
        text_n = text
    if old_string == new_string:
        return "ОШИБКА: old_string и new_string совпадают"
    cnt = text_n.count(old_string)
    if cnt == 0:
        return "ОШИБКА: old_string не найден в файле. Перечитайте файл и скопируйте фрагмент точно, с отступами."
    if cnt > 1 and not replace_all:
        return "ОШИБКА: old_string встречается %d раз. Добавьте контекст или replace_all=true." % cnt
    res = text_n.replace(old_string, new_string) if replace_all else text_n.replace(old_string, new_string, 1)
    if crlf:
        res = res.replace("\n", "\r\n")
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(res)
    return "OK: заменено %d вхождений в %s" % (cnt if replace_all else 1, p)


def t_list_dir(path="."):
    p = _p(path)
    if not os.path.isdir(p):
        return "ОШИБКА: папка не найдена: " + p
    items = []
    for n in sorted(os.listdir(p)):
        full = os.path.join(p, n)
        if os.path.isdir(full):
            items.append(n + "/")
        else:
            try:
                items.append("%s  (%d Б)" % (n, os.path.getsize(full)))
            except OSError:
                items.append(n)
    return "\n".join(items) or "(пусто)"


def _walk(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            yield os.path.join(dp, fn)


def _glob_to_re(pat):
    pat = pat.replace("\\", "/")
    i, out = 0, ""
    while i < len(pat):
        ch = pat[i]
        if pat[i:i + 3] == "**/":
            out += "(?:.*/)?"
            i += 3
            continue
        if pat[i:i + 2] == "**":
            out += ".*"
            i += 2
            continue
        if ch == "*":
            out += "[^/]*"
        elif ch == "?":
            out += "[^/]"
        else:
            out += re.escape(ch)
        i += 1
    return re.compile("^" + out + "$")


def t_glob(pattern, path="."):
    root = _p(path)
    rx = _glob_to_re(pattern)
    res = []
    for f in _walk(root):
        rel = os.path.relpath(f, root).replace("\\", "/")
        if rx.match(rel) or ("/" not in pattern and rx.match(os.path.basename(f))):
            res.append(rel)
            if len(res) >= 500:
                break
    return "\n".join(sorted(res)) or "(ничего не найдено)"


def t_grep(pattern, path=".", include=None, ignore_case=False):
    root = _p(path)
    try:
        rx = re.compile(pattern, re.I if ignore_case else 0)
    except re.error as e:
        return "ОШИБКА: неверное регулярное выражение: %s" % e
    files = [root] if os.path.isfile(root) else _walk(root)
    res = []
    for f in files:
        if include and not fnmatch.fnmatch(os.path.basename(f), include):
            continue
        try:
            if os.path.getsize(f) > 2_000_000:
                continue
            with open(f, "rb") as fh:
                raw = fh.read()
            if b"\0" in raw[:2048]:
                continue
            for n, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                if rx.search(line):
                    rel = os.path.relpath(f, root if os.path.isdir(root) else os.path.dirname(root))
                    res.append("%s:%d: %s" % (rel, n, line.strip()[:300]))
                    if len(res) >= 300:
                        return "\n".join(res) + "\n...[показаны первые 300]"
        except OSError:
            continue
    return "\n".join(res) or "(совпадений нет)"


def t_run_command(command, timeout_seconds=120):
    timeout = max(1, min(int(timeout_seconds or 120), 900))
    if IS_WIN:
        ps = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps]
        kw = {"creationflags": 0x08000000}
    else:
        argv = ["/bin/sh", "-c", command]
        kw = {}
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, cwd=os.getcwd(), **kw)
    except FileNotFoundError as e:
        return "ОШИБКА: не удалось запустить оболочку: %s" % e
    out = []

    def reader():
        for chunk in iter(lambda: p.stdout.read(4096), b""):
            out.append(chunk)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    start = time.time()
    status = None
    while p.poll() is None:
        if CANCEL.is_set():
            p.kill()
            status = "ПРЕРВАНО пользователем"
            break
        if time.time() - start > timeout:
            p.kill()
            status = "ТАЙМАУТ %dс, процесс убит" % timeout
            break
        time.sleep(0.05)
    th.join(2)
    text = b"".join(out).decode("utf-8", "replace")
    res = trunc(text.strip()) or "(нет вывода)"
    if status:
        return res + "\n[" + status + "]"
    return res + "\n[код выхода: %s]" % p.returncode


def t_web_fetch(url):
    if not re.match(r"^https?://", url):
        return "ОШИБКА: нужен http(s) URL"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 AnotherCC"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read(1_500_000).decode("utf-8", "replace")
    except Exception as e:
        return "ОШИБКА: %s" % e
    if "<html" in raw[:2000].lower() or "<body" in raw[:5000].lower():
        raw = re.sub(r"(?is)<(script|style|noscript).*?</\1>", "", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
    return trunc(raw.strip())


TOOLS = {
    "read_file": (t_read_file, False, "Прочитать файл с номерами строк.",
                  {"path": ("string", "Путь к файлу"), "offset": ("integer", "С какой строки (1)"),
                   "limit": ("integer", "Сколько строк (2000)")}, ["path"]),
    "write_file": (t_write_file, True, "Создать или полностью перезаписать файл.",
                   {"path": ("string", "Путь к файлу"), "content": ("string", "Содержимое")}, ["path", "content"]),
    "edit_file": (t_edit_file, True, "Заменить точный фрагмент текста в файле. old_string должен быть уникальным.",
                  {"path": ("string", "Путь"), "old_string": ("string", "Что заменить (точно)"),
                   "new_string": ("string", "На что заменить"),
                   "replace_all": ("boolean", "Заменить все вхождения")}, ["path", "old_string", "new_string"]),
    "list_dir": (t_list_dir, False, "Показать содержимое папки.", {"path": ("string", "Папка (по умолчанию .)")}, []),
    "glob": (t_glob, False, "Найти файлы по маске, например **/*.py.",
             {"pattern": ("string", "Маска"), "path": ("string", "Папка поиска")}, ["pattern"]),
    "grep": (t_grep, False, "Искать регулярное выражение в файлах.",
             {"pattern": ("string", "Регулярка"), "path": ("string", "Файл или папка"),
              "include": ("string", "Маска имени файла, например *.py"),
              "ignore_case": ("boolean", "Без учёта регистра")}, ["pattern"]),
    "run_command": (t_run_command, True,
                    "Выполнить команду в оболочке (%s) в текущей папке." % ("PowerShell" if IS_WIN else "sh"),
                    {"command": ("string", "Команда"),
                     "timeout_seconds": ("integer", "Таймаут в секундах (120, макс. 900)")}, ["command"]),
    "web_fetch": (t_web_fetch, False, "Скачать страницу по URL и вернуть текст.", {"url": ("string", "URL")}, ["url"]),
}


def tool_schemas():
    res = []
    for name, (_, _, desc, params, req) in TOOLS.items():
        props = {k: {"type": v[0], "description": v[1]} for k, v in params.items()}
        res.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}})
    return res


def text_tool_help():
    lines = []
    for name, (_, _, desc, params, req) in TOOLS.items():
        ps = ", ".join("%s%s: %s" % (k, "" if k in req else "?", v[0]) for k, v in params.items())
        lines.append("- %s(%s) - %s" % (name, ps, desc))
    return "\n".join(lines)

# ------------------------------------------------------------------ agent --

CANCEL = threading.Event()


def system_prompt(native):
    parts = [
        "Ты - AnotherCC, ИИ-агент для программирования, работающий в терминале пользователя. "
        "Ты помогаешь с задачами разработки: читаешь и правишь код, запускаешь команды, ищешь ошибки, "
        "пишешь тесты и документацию. Действуй как опытный инженер.",
        "Среда: ОС %s, оболочка %s, рабочая папка: %s, сегодня %s." % (
            platform.system() + " " + platform.release(), "PowerShell" if IS_WIN else "sh",
            os.getcwd(), time.strftime("%Y-%m-%d")),
        "Правила:\n"
        "- Сначала разберись в проекте (list_dir, glob, grep, read_file), потом меняй код.\n"
        "- Для небольших изменений используй edit_file, а не write_file. Перед правкой читай файл.\n"
        "- После изменений проверяй результат: запускай тесты, сборку или сам скрипт.\n"
        "- Работай самостоятельно, пока задача не выполнена; не спрашивай разрешения на каждый шаг.\n"
        "- Если команда упала, прочитай ошибку, исправь причину и повтори.\n"
        "- Не выдумывай содержимое файлов и результаты команд.\n"
        "- Отвечай на языке пользователя, кратко. В конце коротко опиши, что сделано.",
    ]
    for name in ("ANOTHERCC.md", "CLAUDE.md", "AGENTS.md"):
        p = os.path.join(os.getcwd(), name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    parts.append("Инструкции проекта (%s):\n%s" % (name, f.read()[:8000]))
            except OSError:
                pass
            break
    if not native:
        parts.append(
            "У тебя есть инструменты. Чтобы вызвать инструмент, ответь блоком ровно такого вида и больше ничего "
            "после него не пиши:\n```tool\n{\"name\": \"имя\", \"arguments\": {...}}\n```\n"
            "Можно вызывать несколько инструментов подряд - несколько блоков. Результат придёт в следующем "
            "сообщении. Когда задача выполнена, отвечай обычным текстом без блоков tool.\n"
            "Инструменты:\n" + text_tool_help())
    return "\n\n".join(parts)


def parse_text_tools(text):
    calls = []
    for m in re.finditer(r"```tool\s*\n(.*?)```", text, re.S):
        try:
            obj = json.loads(m.group(1).strip())
            calls.append((obj.get("name", ""), obj.get("arguments", {}) or {}))
        except Exception:
            continue
    return calls


def preview(name, args):
    if name == "run_command":
        return args.get("command", "")
    if name == "write_file":
        content = args.get("content", "")
        return "%s (%d симв.)\n%s" % (args.get("path"), len(content), dim(trunc(content, 600)))
    if name == "edit_file":
        return "%s\n%s\n%s" % (args.get("path"),
                               red("- " + trunc(args.get("old_string", ""), 400).replace("\n", "\n- ")),
                               green("+ " + trunc(args.get("new_string", ""), 400).replace("\n", "\n+ ")))
    return json.dumps(args, ensure_ascii=False)[:200]


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = LLM(cfg)
        self.auto = False
        self.always = set()
        self.messages = []
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": system_prompt(self.llm.native_tools)}]

    def confirm(self, name, args):
        if self.auto or name in self.always:
            return True
        print(yellow("\n? Разрешить %s:" % name))
        print(preview(name, args))
        while True:
            a = ask(yellow("[y] да / [a] всегда в этой сессии / [n] нет: ")).strip().lower()
            if a in ("y", "д", "да", ""):
                return True
            if a in ("a", "ф", "всегда"):
                self.always.add(name)
                return True
            if a in ("n", "н", "нет"):
                return False

    def run_tool(self, name, args):
        if name not in TOOLS:
            return "ОШИБКА: неизвестный инструмент " + str(name)
        fn, needs_confirm, _, params, req = TOOLS[name]
        if not isinstance(args, dict):
            return "ОШИБКА: аргументы должны быть JSON-объектом"
        for r in req:
            if r not in args:
                return "ОШИБКА: не указан обязательный аргумент '%s'" % r
        args = {k: v for k, v in args.items() if k in params}
        if needs_confirm and not self.confirm(name, args):
            return "Пользователь отклонил это действие. Предложи другой путь или спроси, что делать."
        try:
            return str(fn(**args))
        except Exception as e:
            return "ОШИБКА при выполнении: %s: %s" % (type(e).__name__, e)

    def compact(self):
        total = sum(len(str(m.get("content") or "")) for m in self.messages)
        if total <= MAX_HISTORY_CHARS:
            return
        for m in self.messages[1:-8]:
            if m["role"] in ("tool", "user") and m.get("content") and len(m["content"]) > 1500 \
                    and not m["content"].startswith("[сжато]"):
                m["content"] = "[сжато] " + m["content"][:600] + "\n...[старый вывод удалён для экономии контекста]"
            total = sum(len(str(x.get("content") or "")) for x in self.messages)
            if total <= MAX_HISTORY_CHARS:
                return

    def turn(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        CANCEL.clear()
        steps = 0
        while True:
            steps += 1
            if steps > 60:
                print(yellow("Достигнут лимит шагов (60). Напишите «продолжай», чтобы идти дальше."))
                return
            self.compact()
            sp = self.messages[0]
            sp["content"] = system_prompt(self.llm.native_tools)
            sys.stdout.write(dim("  думаю..."))
            sys.stdout.flush()
            try:
                res = self.llm.chat(self.messages, tool_schemas())
            except Cancelled:
                print("\r" + " " * 20 + "\r" + yellow("Прервано."))
                if self.messages[-1]["role"] == "user":
                    self.messages.pop()
                return
            except APIError as e:
                print("\r" + " " * 20 + "\r" + red("Ошибка API: %s" % e))
                if e.status in (401, 403):
                    print(yellow("Проверьте ключ: команда /setup"))
                elif e.status in (400, 404):
                    print(yellow("Возможно, модель недоступна. Смотрите /models и меняйте через /model <имя>"))
                if self.messages[-1]["role"] == "user":
                    self.messages.pop()
                return
            print("\r" + " " * 20 + "\r", end="")
            msg = res["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning_stripped = re.sub(r"(?is)<think>.*?</think>", "", content).strip()
            native_calls = msg.get("tool_calls") or []
            if native_calls and self.llm.native_tools:
                if reasoning_stripped:
                    print(reasoning_stripped)
                self.messages.append({"role": "assistant", "content": content or None, "tool_calls": native_calls})
                for tc in native_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = None
                    print(cyan("● %s" % name) + dim(" " + (json.dumps(args, ensure_ascii=False)[:120] if args else "")))
                    out = self.run_tool(name, args) if args is not None else "ОШИБКА: невалидный JSON в аргументах"
                    print(dim("  " + trunc(out, 300).replace("\n", "\n  ")))
                    self.messages.append({"role": "tool", "tool_call_id": tc.get("id", name), "content": trunc(out)})
                    if CANCEL.is_set():
                        print(yellow("Прервано."))
                        return
                continue
            text_calls = [] if self.llm.native_tools else parse_text_tools(content)
            if text_calls:
                visible = re.sub(r"```tool\s*\n.*?```", "", reasoning_stripped, flags=re.S).strip()
                if visible:
                    print(visible)
                self.messages.append({"role": "assistant", "content": content})
                results = []
                for name, args in text_calls:
                    print(cyan("● %s" % name) + dim(" " + json.dumps(args, ensure_ascii=False)[:120]))
                    out = self.run_tool(name, args)
                    print(dim("  " + trunc(out, 300).replace("\n", "\n  ")))
                    results.append("Результат %s:\n%s" % (name, trunc(out)))
                    if CANCEL.is_set():
                        print(yellow("Прервано."))
                        return
                self.messages.append({"role": "user", "content": "\n\n".join(results)})
                continue
            self.messages.append({"role": "assistant", "content": content})
            print(reasoning_stripped or dim("(пустой ответ)"))
            return

# -------------------------------------------------------------------- CLI --

HELP = """Команды:
  /help            эта справка
  /setup           заново выбрать провайдера и ключ
  /models          список моделей провайдера (бесплатные помечены *)
  /model <имя>     сменить модель
  /auto            вкл/выкл автоподтверждение правок и команд
  /clear           начать новый диалог
  /cd <папка>      сменить рабочую папку
  /exit            выход
Ввод в несколько строк: начните и закончите сообщение строкой \"\"\"
Ctrl+C во время работы - прервать текущую задачу."""


def read_user():
    line = ask(green("\n> "))
    if line.strip() == '"""':
        buf = []
        while True:
            l = ask("")
            if l.strip() == '"""':
                break
            buf.append(l)
        return "\n".join(buf)
    return line


def main():
    _enable_console()
    args = sys.argv[1:]
    cfg = load_config()
    auto = False
    prompt_once = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-y", "--yes", "--auto"):
            auto = True
        elif a in ("-m", "--model") and i + 1 < len(args):
            cfg["model"] = args[i + 1]
            i += 1
        elif a in ("-p", "--prompt") and i + 1 < len(args):
            prompt_once = args[i + 1]
            i += 1
        elif a in ("-h", "--help"):
            print("AnotherCC %s\nИспользование: AnotherCC [-y] [-m модель] [-p \"задача\"]\n\n%s" % (VERSION, HELP))
            return
        elif a == "--setup":
            cfg = setup_wizard(cfg)
        i += 1

    env_key = os.environ.get("ANOTHERCC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if env_key and not cfg.get("api_key"):
        cfg["api_key"] = env_key
        cfg.setdefault("base_url", "https://openrouter.ai/api/v1")
        cfg.setdefault("model", "openrouter/free")
    if not cfg.get("base_url") or not cfg.get("model"):
        cfg = setup_wizard(cfg)

    agent = Agent(cfg)
    agent.auto = auto

    print(bold(cyan("AnotherCC %s" % VERSION)) + dim("  модель: %s" % cfg["model"]))
    print(dim("Папка: %s   /help - команды" % os.getcwd()))
    if auto:
        print(yellow("Автоподтверждение ВКЛ: правки и команды выполняются без вопросов."))

    def on_sigint(sig, frm):
        if agent_busy[0]:
            CANCEL.set()
        else:
            print("\nВыход. До встречи!")
            os._exit(0)
    agent_busy = [False]
    import signal
    signal.signal(signal.SIGINT, on_sigint)

    if prompt_once:
        agent_busy[0] = True
        agent.turn(prompt_once)
        return

    while True:
        agent_busy[0] = False
        try:
            text = read_user().strip()
        except KeyboardInterrupt:
            continue
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, rest = text.partition(" ")
            rest = rest.strip()
            if cmd in ("/exit", "/quit", "/q"):
                return
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/clear":
                agent.reset()
                print(dim("Диалог очищен."))
            elif cmd == "/auto":
                agent.auto = not agent.auto
                print(yellow("Автоподтверждение: %s" % ("ВКЛ" if agent.auto else "выкл")))
            elif cmd == "/setup":
                cfg = setup_wizard(cfg)
                agent = Agent(cfg)
            elif cmd == "/model":
                if rest:
                    cfg["model"] = rest
                    save_config(cfg)
                    agent.llm.cfg = cfg
                    agent.llm.native_tools = True
                    print(green("Модель: " + rest))
                else:
                    print("Текущая модель: " + cfg["model"])
            elif cmd == "/models":
                try:
                    ms = agent.llm.list_models()
                    ms.sort(key=lambda x: (not x[1], x[0] or ""))
                    for mid, free in ms[:80]:
                        print(("* " if free else "  ") + str(mid))
                    if len(ms) > 80:
                        print(dim("...и ещё %d" % (len(ms) - 80)))
                    print(dim("Выбрать: /model <имя>"))
                except Exception as e:
                    print(red("Не удалось получить список: %s" % e))
            elif cmd == "/cd":
                try:
                    os.chdir(_p(rest))
                    print(dim("Папка: " + os.getcwd()))
                    agent.reset()
                except Exception as e:
                    print(red(str(e)))
            else:
                print(red("Неизвестная команда. /help"))
            continue
        agent_busy[0] = True
        try:
            agent.turn(text)
        except KeyboardInterrupt:
            print(yellow("Прервано."))
        agent_busy[0] = False


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        if IS_WIN and sys.stdin and sys.stdin.isatty():
            try:
                input("\nНажмите Enter для выхода...")
            except Exception:
                pass
        sys.exit(1)
