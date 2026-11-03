import asyncio
import os
import re
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.ollama import OllamaChatClient  # правильный импорт

# ========== КОНФИГУРАЦИЯ ==========
PLANNER_MODEL = "deepseek-r1:8b"
EXECUTOR_MODEL = "lfm3:latest"
MAX_STEPS = 5

# ========== ИНСТРУМЕНТЫ ДЛЯ ИСПОЛНИТЕЛЯ ==========
@tool(approval_mode="never_require")
def read_knowledge_base(
    topic_name: Annotated[str, "Тема документации: 'llama_cpp' или 'hardware_requirements'"]
) -> str:
    """
    Читает проверенную техническую документацию по указанной теме.
    Обязательно используй этот инструмент перед написанием инструкций,
    команд запуска или описанием требований к железу.
    """
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
    """
    Сохраняет переданный текст в локальный файл в текущей папке проекта.
    Используй этот инструмент, когда нужно записать готовый раздел статьи,
    код или отчёт на диск.
    """
    try:
        safe_filename = os.path.basename(filename)
        with open(safe_filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Успех: данные записаны в файл {safe_filename}."
    except Exception as e:
        return f"Ошибка при записи файла: {str(e)}"

# ========== СИСТЕМНЫЕ ПРОМПТЫ ==========
PLANNER_SYSTEM_PROMPT = (
    "Ты — Главный Архитектор и Планировщик (Team Lead). Твоя цель — управлять исполнителем "
    "для достижения глобальной цели пользователя.\n\n"
    "ПРАВИЛА:\n"
    "1. Анализируй всю историю выполнения: что уже сделано, что осталось.\n"
    "2. Формируй строго ОДИН следующий логический шаг. Не пытайся решить всё сразу.\n"
    "3. Если предыдущий результат исполнителя содержит ошибку, непонимание или пустой текст — "
    "переформируй задачу, дай подсказку или заставь переделать работу.\n"
    "4. Когда все критерии глобальной цели достигнуты, выставь status=done.\n\n"
    "ФОРМАТ ОТВЕТА (ЖЁСТКИЙ КРИТЕРИЙ):\n"
    "Ты должен вернуть ответ СТРОГО в формате Key-Value (HTTP form-data), "
    "каждая пара на новой строке. Никакого лишнего текста вне ключей.\n\n"
    "task_name=техническое_название_шага (латиницей, без пробелов)\n"
    "payload=подробная инструкция для исполнителя. Если нужны переносы строк, "
    "используй \\n, чтобы не сломать парсер.\n"
    "status=continue (или status=done, если цель полностью достигнута)\n\n"
    "ВАЖНО: Если задача требует описания установки, команд запуска или технических "
    "характеристик, в payload ЖЁСТКО потребуй от исполнителя использовать инструмент "
    "'read_knowledge_base' с нужной темой (например, 'llama_cpp' или 'hardware_requirements').\n"
    "Не используй символ '=' внутри текста payload — заменяй на ':' или слова."
)

EXECUTOR_SYSTEM_PROMPT = (
    "Ты — линейный исполнитель. Чётко выполняй инструкцию из payload.\n"
    "Тебе доступны инструменты:\n"
    "- read_knowledge_base: читает проверенную документацию по теме.\n"
    "- save_to_file: сохраняет результат в файл.\n"
    "Если в payload сказано использовать инструмент — ОБЯЗАТЕЛЬНО вызови его и "
    "строй ответ только на основе полученных данных. Не придумывай команды, форматы "
    "или характеристики из головы.\n"
    "Выдавай только результат, без мета-комментариев и вступлений."
)

# ========== СОЗДАНИЕ ЛОКАЛЬНОЙ БАЗЫ ЗНАНИЙ ==========
def ensure_knowledge_base():
    """Создаёт файлы с точными техническими данными, чтобы LFM3 не галлюцинировал."""
    docs = {
        "llama_cpp_doc.txt": (
            "Документация llama.cpp (актуально на 2026 год):\n"
            "1. Формат моделей: строго GGUF (файлы .gguf). Формат GGML больше не поддерживается.\n"
            "2. Быстрая установка одной командой:\n"
            "   Windows (PowerShell): irm https://llama.app | iex\n"
            "   Linux/macOS: curl -LsSf https://llama.app | sh\n"
            "3. Запуск модели из Hugging Face:\n"
            "   llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF\n"
            "4. Запуск локального OpenAI-совместимого сервера:\n"
            "   llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF\n"
        ),
        "hardware_requirements_doc.txt": (
            "Требования к железу для локальных LLM (актуально на 2026 год):\n"
            "- Для моделей до 1 млрд параметров: 8 ГБ RAM, GPU с 4 ГБ VRAM (опционально).\n"
            "- Для моделей 3–7 млрд параметров: 16 ГБ RAM, GPU с 8 ГБ VRAM (рекомендуется).\n"
            "- Для моделей 13B+: 32 ГБ RAM, GPU с 12–16 ГБ VRAM.\n"
            "- Квантование (4-bit) снижает требования к VRAM примерно в 2–3 раза.\n"
            "- CPU: достаточно 4–8 ядер; GPU ускоряет инференс, но не обязателен.\n"
            "Не используй устаревшие форматы вроде .ggmlv3 и несуществующие параметры CLI.\n"
        )
    }
    for filename, content in docs.items():
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📚 Создана база знаний: {filename}")

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def parse_key_value(text: str) -> dict:
    """Парсит Key-Value формат (HTTP form-data) в словарь."""
    parsed = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            key, val = line.split("=", 1)
            parsed[key.strip()] = val.strip()
    return parsed

def extract_text(result) -> str:
    """Извлекает финальный текст из ответа агента MAF."""
    if hasattr(result, "text"):
        return result.text or ""
    return str(result)

# ========== ГЛАВНЫЙ ЦИКЛ ОРКЕСТРАЦИИ ==========
async def main():
    ensure_knowledge_base()

    # Клиенты Ollama
    planner_client = OllamaChatClient(model=PLANNER_MODEL)
    executor_client = OllamaChatClient(model=EXECUTOR_MODEL)

    # Планировщик с нативной поддержкой рассуждений
    planner_agent = Agent(
        client=planner_client,
        name="PlannerAgent",
        instructions=PLANNER_SYSTEM_PROMPT,
        default_options={"think": True},  # DeepSeek-R1 будет рассуждать
    )

    # Исполнитель с инструментами
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
        print(f"\n🧠 [Шаг {step}/{MAX_STEPS}] Планировщик ({PLANNER_MODEL}) думает...")

        planner_result = await planner_agent.run(current_context)
        raw_text = extract_text(planner_result)

        # Нативный reasoning уже отделён, но на всякий случай очищаем остатки <think>
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        task_data = parse_key_value(clean_text)
        task_name = task_data.get("task_name", "UnknownTask")
        payload = task_data.get("payload", "").replace("\\n", "\n")
        status = task_data.get("status", "continue")

        if not payload:
            print("⚠️ Планировщик вернул пустой payload или нарушил формат.")
            print("Сырой ответ:", clean_text[:500])
            break

        print(f"📋 Задача: {task_name}")
        print(f"📦 Payload: {payload[:300]}{'...' if len(payload) > 300 else ''}")

        if status == "done":
            print("\n🎉 [УСПЕХ] Планировщик сообщил, что глобальная цель достигнута.")
            break

        print(f"⚡ Исполнитель ({EXECUTOR_MODEL}) выполняет...")
        executor_result = await executor_agent.run(payload)
        execution_output = extract_text(executor_result)

        print(f"✅ Результат:\n{execution_output}\n" + "-" * 60)

        # Накапливаем историю для следующего шага
        current_context += (
            f"\n\n[Шаг {step}: {task_name}]\n"
            f"Инструкция: {payload}\n"
            f"Фактический результат: {execution_output}\n"
            f"Проанализируй и реши, что делать дальше."
        )

        # Дополнительная проверка: если исполнитель явно сообщил об ошибке
        if "Ошибка" in execution_output or "error" in execution_output.lower():
            current_context += "\nВНИМАНИЕ: исполнитель вернул ошибку. Скорректируй задачу или потребуй переделать."

    else:
        print(f"\n⏳ Достигнут лимит шагов ({MAX_STEPS}), но status=done не получен.")

    print("\n🏁 Работа завершена.")

if __name__ == "__main__":
    asyncio.run(main())