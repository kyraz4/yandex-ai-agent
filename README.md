# yandex-ai-agent


## 1. Требования

- **Python 3.12** или **3.13**. Не используйте Python 3.15 — для него ещё нет стабильного `torch`, и `sentence-transformers` не установится.
- **Ollama** для macOS: [ollama.com/download/mac](https://ollama.com/download/mac) или `brew install ollama`.

Проверьте версию Python:

```bash
python3.12 --version
```

Если 3.12 нет:

```bash
brew install python@3.12
```

Установка зависимостей:

```bash
cd /path/to/yandex-ai-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txtrequirements.txt
```

Установка Ollama и модели:

```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:1.5b
```

Проверка работы:

```bash
curl http://localhost:11434/api/tags | grep qwen2.5
```

Запуск бота: 

```bash
cd knowledge_base

# (Опционально) Пересобрать индекс, если vector_index/ нет в репо
python build_index.py

# Убедиться, что Ollama запущен и модель скачана
ollama list | grep qwen2.5

# Запустить бота
python rag_bot.py
```

### Запуск через Docker (альтернатива локальному venv)

```bash
docker compose build
docker compose up -d ollama
docker exec -it ollama ollama pull qwen2.5:1.5b
docker compose run --rm bot
```
