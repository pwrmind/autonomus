import asyncio
import os
import re
from agent_framework import Agent
from agent_framework.ollama import OllamaChatClient

# Конфигурация локальных моделей через переменные окружения, как в примерах MAF
os.environ["OLLAMA_MODEL"] = "deepseek-r1:8b"  # Модель планировщика по умолчанию

async def main():
    # Инициализируем нативные клиенты Ollama согласно спецификации MAF
    # Клиент автоматически подтянет OLLAMA_HOST (http://localhost:11434)
    planner_client = OllamaChatClient(model="deepseek-r1:8b")
    executor_client = OllamaChatClient(model="lfm3:latest")

    # Создаем Агента-Планировщика
    planner_agent = Agent(
        client=planner_client,
        name="PlannerAgent",
        instructions=(
            "Ты — Главный Планировщик. Твоя задача — разбить глобальную цель пользователя на пошаговые задачи. "
            "На каждый запрос ты должен выдать ОДНУ следующую конкретную подзадачу для исполнителя. "
            "Ответ ты ОБЯЗАН предоставить строго в формате Key-Value (как HTTP form-data), разделяя пары новой строкой.\n\n"
            "Формат ответа:\n"
            "task_name=название_действия\n"
            "payload=текст_задания_для_исполнителя\n"
            "status=continue (или status=done, если цель полностью достигнута)"
        )
    )

    # Создаем Агента-Исполнителя
    executor_agent = Agent(
        client=executor_client,
        name="ExecutorAgent",
        instructions="Ты — линейный исполнитель. Четко выполняй то, что написано в переданном payload, без лишней воды."
    )

    # Исходная цель
    global_goal = "Составь план для технической статьи про локальные LLM и подробно распиши введение."
    
    print(f"🎯 Глобальная цель: {global_goal}\n" + "="*60)
    
    current_context = f"Глобальная цель: {global_goal}"
    max_steps = 3

    for step in range(max_steps):
        print(f"\n🧠 [Шаг {step + 1}] Планировщик генерирует задачу...")
        
        # Вызов агента возвращает объект ответа, текст лежит в .text
        planner_response = await planner_agent.run(current_context)
        content = planner_response.text if hasattr(planner_response, 'text') else str(planner_response)
        
        # Надежно отсекаем теги рассуждений <think>...</think> от deepseek-r1
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Парсим HTTP-подобный Key-Value формат в словарь
        parsed_task = {}
        for line in clean_content.split('\n'):
            if '=' in line:
                key, val = line.split('=', 1)
                parsed_task[key.strip()] = val.strip()
        
        task_name = parsed_task.get('task_name', 'UnknownTask')
        payload = parsed_task.get('payload', '')
        status = parsed_task.get('status', 'continue')
        
        print(f"📋 Сгенерирована задача: {task_name}")
        print(f"📦 Данные (payload): {payload}")
        
        if not payload:
            print("⚠️ Ошибка: Планировщик нарушил формат Key-Value. Текст ответа:")
            print(clean_content)
            break
            
        if status == "done":
            print("🎉 Планировщик сообщил, что вся цель успешно достигнута!")
            break
            
        print(f"⚡ Отправка задачи исполнителю {executor_agent.name}...")
        
        # Передаем payload в LFM3
        executor_response = await executor_agent.run(payload)
        execution_result = executor_response.text if hasattr(executor_response, 'text') else str(executor_response)
        
        print(f"✅ Результат выполнения:\n{execution_result}")
        print("-" * 60)
        
        # Обновляем лог событий для следующего шага
        current_context += (
            f"\n\n[Лог выполнения шага {step + 1}]:\n"
            f"Выполненная задача: {task_name}\n"
            f"Результат работы исполнителя: {execution_result}\n"
            f"Сделай следующий шаг на основе этого результата."
        )

if __name__ == "__main__":
    asyncio.run(main())
