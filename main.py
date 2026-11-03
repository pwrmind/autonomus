import asyncio
import os
import re
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.ollama import OllamaChatClient

# ========== КОНФИГУРАЦИЯ ==========
PLANNER_MODEL = "deepseek-r1:8b"
EXECUTOR_MODEL = "lfm3:latest"
MAX_STEPS = 5

DEBUG_STREAM = False
REASONING_FLUSH_SIZE = 120
ANSWER_FLUSH_SIZE = 60
QUIET_REASONING = False

# ========== ИНСТРУМЕНТЫ ==========
@tool(approval_mode="never_require")
def read_knowledge_base(
    topic_name: Annotated[str, "Тема документации: 'llama_cpp' или 'hardware_requirements'"]
) -> str:
    """Читает проверенную техническую документацию по указанной теме."""
    filename = f"{topic_name}_doc.txt"
    if not os.path.exists(filename):
        return f"Ошибка: документация по теме '{topic_name}' не найдена."
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


@tool(approval_mode="never_require")
def save_to_file(
    filename: Annotated[str, "Имя файла для сохранения (без пути)"],
    content: Annotated[str, "Текст, который нужно записать в файл"]
) -> str:
    """Сохраняет переданный текст в локальный файл в текущей папке проекта."""
    try:
        safe_filename = os.path.basename(filename)
        with open(safe_filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Успех: данные записаны в файл {safe_filename}."
    except Exception as e:
        return f"Ошибка при записи файла: {str(e)}"


# ========== ПРОМПТЫ ==========
PLANNER_SYSTEM_PROMPT = (
    "Ты — Главный Архитектор и Планировщик (Team Lead). Управляй исполнителем для достижения цели.\n\n"
    "ПРАВИЛА ПЛАНИРОВАНИЯ:\n"
    "1. Анализируй историю: что сделано, что осталось.\n"
    "2. Формируй строго ОДИН следующий шаг.\n"
    "3. Если результат исполнителя плох — переформируй задачу или потребуй переделать.\n"
    "4. Когда цель достигнута — status=done.\n\n"
    "ФОРМАТ ОТВЕТА (ЖЁСТКО, без отклонений):\n"
    "Ответ — ТОЛЬКО три строки. Разделитель между ключом и значением — знак РАВНО '='.\n"
    "После 'task_name', 'payload' и 'status' ВСЕГДА ставь '=' (не двоеточие!).\n\n"
    "ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:\n"
    "task_name=write_structure\n"
    "payload=Создай HTML-шаблон с разделами:\\n1. Заголовок\\n2. Описание услуг\\n3. Контакты\n"
    "status=continue\n\n"
    "ТРЕБОВАНИЯ К ПОЛЯМ:\n"
    "- task_name: короткое имя латиницей, без пробелов (например: write_structure).\n"
    "- payload: инструкция для исполнителя. Значение payload — ОДНА строка. "
    "Реальные переносы строк внутри payload запрещены — используй \\n (обратный слэш + n). "
    "ВНУТРИ значения payload не используй '=' — заменяй на ':' или слова.\n"
    "- status: ровно одно слово — continue или done, без пояснений.\n\n"
    "Если нужна документация или технические характеристики — в payload потребуй "
    "использовать инструмент read_knowledge_base с нужной темой."
)

EXECUTOR_SYSTEM_PROMPT = (
    "Ты — линейный исполнитель. Чётко выполняй payload.\n"
    "Доступные инструменты:\n"
    "- read_knowledge_base: читает проверенную документацию.\n"
    "- save_to_file: сохраняет результат в файл.\n"
    "Если в payload сказано использовать инструмент — ОБЯЗАТЕЛЬНО вызови его и "
    "строй ответ только на основе полученных данных.\n"
    "Выдавай только результат, без мета-комментариев."
)


# ========== БАЗА ЗНАНИЙ ==========
def ensure_knowledge_base():
    docs = {
        "llama_cpp_doc.txt": (
            "Документация llama.cpp (2026):\n"
            "1. Формат моделей: строго GGUF (.gguf). GGML больше не поддерживается.\n"
            "2. Установка:\n"
            "   Windows: irm https://llama.app | iex\n"
            "   Linux/macOS: curl -LsSf https://llama.app | sh\n"
            "3. Запуск модели:\n"
            "   llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF\n"
            "4. OpenAI-совместимый сервер:\n"
            "   llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF\n"
        ),
        "hardware_requirements_doc.txt": (
            "Требования к железу для локальных LLM (2026):\n"
            "- до 1B параметров: 8 ГБ RAM, GPU 4 ГБ VRAM (опционально).\n"
            "- 3–7B: 16 ГБ RAM, GPU 8 ГБ VRAM (рекомендуется).\n"
            "- 13B+: 32 ГБ RAM, GPU 12–16 ГБ VRAM.\n"
            "- Квантование 4-bit снижает VRAM в 2–3 раза.\n"
            "- CPU: 4–8 ядер достаточно; GPU ускоряет, но не обязателен.\n"
            "Не используй .ggmlv3 и несуществующие параметры CLI.\n"
        ),
    }
    for filename, content in docs.items():
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📚 Создана база знаний: {filename}")


# ========== РОБАСТНЫЙ ПАРСЕР ==========
def parse_planner_output(text: str) -> dict:
    """
    Устойчивый парсер Key-Value вывода планировщика.
    Принимает и '=', и ':' как разделитель ключа и значения.
    Собирает многострочный payload до ближайшего 'status[:=]' или конца текста.
    Разэкранирует '\\n', если модель использовала escape-последовательности.
    """
    # Убираем возможные остатки <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    result: dict = {}

    # task_name: значение до конца строки
    m = re.search(
        r"^[ \t]*task_name[ \t]*[:=][ \t]*(.+?)[ \t]*$",
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        result["task_name"] = m.group(1).strip()

    # status: значение до конца строки
    m = re.search(
        r"^[ \t]*status[ \t]*[:=][ \t]*(.+?)[ \t]*$",
        text, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        result["status"] = m.group(1).strip().lower()

    # payload: всё от 'payload[:=]' до ближайшего '^status[:=]' или конца текста
    m = re.search(
        r"^[ \t]*payload[ \t]*[:=][ \t]*(.*?)(?=^[ \t]*status[ \t]*[:=]|\Z)",
        text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if m:
        payload = m.group(1).strip()
        # Если модель экранировала переносы как \n, а реальных переносов нет — разэкранируем
        if "\\n" in payload and "\n" not in payload:
            payload = payload.replace("\\n", "\n")
        result["payload"] = payload

    return result


# ========== БУФЕРИЗОВАННЫЙ СТРИМИНГ ==========
class StreamBuffer:
    def __init__(self, label: str, flush_size: int, quiet: bool = False):
        self.label = label
        self.flush_size = flush_size
        self.quiet = quiet
        self._buffer: list[str] = []
        self._printed_header = False
        self.total_chars = 0

    def add(self, text: str):
        if not text:
            return
        self.total_chars += len(text)
        if self.quiet:
            return
        self._buffer.append(text)
        if sum(len(x) for x in self._buffer) >= self.flush_size:
            self.flush()

    def flush(self):
        if self.quiet or not self._buffer:
            return
        chunk = "".join(self._buffer)
        self._buffer.clear()
        if not self._printed_header:
            print(f"\n{self.label} ", end="", flush=True)
            self._printed_header = True
        print(chunk, end="", flush=True)

    def finish(self):
        self.flush()
        if not self.quiet and self._printed_header:
            print()


def _iter_contents(update):
    contents = getattr(update, "contents", None)
    if contents is None:
        return []
    return contents if isinstance(contents, (list, tuple)) else [contents]


async def run_planner_streaming(agent: Agent, context: str) -> tuple[str, str]:
    print("\n🧠 [Планировщик] Поток рассуждений и ответа:")
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []

    reasoning_buf = StreamBuffer("[THINK]", REASONING_FLUSH_SIZE, QUIET_REASONING)
    answer_buf = StreamBuffer("[ANSWER]", ANSWER_FLUSH_SIZE, False)

    response_stream = agent.run(context, stream=True)
    got_any = False

    async for update in response_stream:
        got_any = True
        for content in _iter_contents(update):
            ctype = getattr(content, "type", None)
            ctext = getattr(content, "text", None) or ""

            if ctype == "text_reasoning" and ctext:
                reasoning_parts.append(ctext)
                reasoning_buf.add(ctext)
                if QUIET_REASONING and reasoning_buf.total_chars % 200 < len(ctext):
                    print(".", end="", flush=True)
            elif ctype == "text" and ctext:
                answer_parts.append(ctext)
                answer_buf.add(ctext)

    reasoning_buf.finish()
    answer_buf.finish()

    if not got_any:
        print("⚠️ Стрим не вернул ни одного чанка.")

    return "".join(reasoning_parts), "".join(answer_parts)


async def run_executor_streaming(agent: Agent, payload: str) -> str:
    print("⚡ [Исполнитель] Потоковый вывод:")
    result_parts: list[str] = []
    answer_buf = StreamBuffer("[OUT]", ANSWER_FLUSH_SIZE, False)

    response_stream = agent.run(payload, stream=True)
    got_any = False

    async for update in response_stream:
        got_any = True
        fc = getattr(update, "function_call", None)
        if fc:
            name = getattr(fc, "name", "?")
            print(f"\n🔧 Вызов инструмента: {name}")

        for content in _iter_contents(update):
            ctype = getattr(content, "type", None)
            ctext = getattr(content, "text", None) or ""
            if ctype == "text" and ctext:
                result_parts.append(ctext)
                answer_buf.add(ctext)

    answer_buf.finish()
    if not got_any:
        print("⚠️ Стрим исполнителя не вернул ни одного чанка.")
    return "".join(result_parts)


# ========== ГЛАВНЫЙ ЦИКЛ ==========
async def main():
    ensure_knowledge_base()

    planner_client = OllamaChatClient(model=PLANNER_MODEL)
    executor_client = OllamaChatClient(model=EXECUTOR_MODEL)

    planner_agent = Agent(
        client=planner_client,
        name="PlannerAgent",
        instructions=PLANNER_SYSTEM_PROMPT,
        default_options={"think": True},
    )
    executor_agent = Agent(
        client=executor_client,
        name="ExecutorAgent",
        instructions=EXECUTOR_SYSTEM_PROMPT,
        tools=[read_knowledge_base, save_to_file],
    )

    print("🤖 Автономный мультиагент готов.")
    global_goal = input("🎯 Введите глобальную задачу: ").strip()
    if not global_goal:
        print("❌ Пустая задача. Выход.")
        return

    print(f"\n🚀 Цель: {global_goal}\n" + "=" * 60)
    current_context = f"Глобальная цель пользователя: {global_goal}"

    for step in range(1, MAX_STEPS + 1):
        print(f"\n{'=' * 60}\nШАГ {step} / {MAX_STEPS}")

        _, raw_answer = await run_planner_streaming(planner_agent, current_context)
        task_data = parse_planner_output(raw_answer)

        task_name = task_data.get("task_name", "UnknownTask")
        payload = task_data.get("payload", "").strip()
        status = task_data.get("status", "continue")

        if not payload:
            print("⚠️ Планировщик вернул пустой payload или нарушил формат.")
            print("Сырой ответ (первые 800 символов):")
            print(raw_answer[:800])
            break

        print(f"\n📋 Задача: {task_name}")
        preview = payload[:300] + ("..." if len(payload) > 300 else "")
        print(f"📦 Payload: {preview}")

        if status == "done":
            print("\n🎉 [УСПЕХ] Планировщик сообщил, что глобальная цель достигнута.")
            break

        execution_output = await run_executor_streaming(executor_agent, payload)
        print("-" * 60)

        current_context += (
            f"\n\n[Шаг {step}: {task_name}]\n"
            f"Инструкция: {payload}\n"
            f"Фактический результат: {execution_output}\n"
            f"Проанализируй и реши, что делать дальше."
        )

        if "Ошибка" in execution_output or "error" in execution_output.lower():
            current_context += (
                "\nВНИМАНИЕ: исполнитель вернул ошибку. "
                "Скорректируй задачу или потребуй переделать."
            )
    else:
        print(f"\n⏳ Достигнут лимит шагов ({MAX_STEPS}), но status=done не получен.")

    print("\n🏁 Работа завершена.")


if __name__ == "__main__":
    asyncio.run(main())