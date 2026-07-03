import os
from fastapi import FastAPI, HTTPException, Security, status, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="Dynamic AI Text Detector API")

# Конфигурация путей и безопасности
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
API_KEY = os.getenv("API_KEY", "my_secure_secret_token_2026")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Глобальные переменные для активной модели
current_model_name = None
model = None
tokenizer = None
device = "cpu"

# Статусы фонового скачивания моделей
download_tasks = {}


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Неверный или отсутствующий API Ключ"
    )


def download_model_worker(repo_id: str, folder_name: str):
    """Фоновая функция для скачивания модели с Hugging Face"""
    target_path = os.path.join(MODELS_DIR, folder_name)
    try:
        download_tasks[repo_id] = "Скачивание началось..."

        # Скачиваем и сохраняем локально
        temp_tokenizer = AutoTokenizer.from_pretrained(repo_id)
        temp_model = AutoModelForSequenceClassification.from_pretrained(repo_id)

        temp_tokenizer.save_pretrained(target_path)
        temp_model.save_pretrained(target_path)

        download_tasks[repo_id] = "Успешно скачано"
    except Exception as e:
        download_tasks[repo_id] = f"Ошибка: {str(e)}"


def load_model_into_memory(folder_name: str) -> bool:
    """Вспомогательная функция для переключения модели в ОЗУ (CPU)"""
    global model, tokenizer, current_model_name
    target_path = os.path.join(MODELS_DIR, folder_name)

    if not os.path.exists(target_path):
        return False

    try:
        # Явно освобождаем оперативную память от старой модели
        global model, tokenizer
        model = None
        tokenizer = None

        # Загружаем модель сразу на CPU
        tokenizer = AutoTokenizer.from_pretrained(target_path)
        model = AutoModelForSequenceClassification.from_pretrained(target_path)
        model.to(device)  # device всегда "cpu"
        model.eval()

        current_model_name = folder_name
        return True
    except Exception as e:
        print(f"Ошибка активации модели {folder_name}: {e}")
        return False


# --- МАРШРУТЫ API ---

class DownloadRequest(BaseModel):
    repo_id: str  # Пример: "desklib/ai-text-detector-v1.01"


@app.post("/api/models/download")
async def download_model(request: DownloadRequest, background_tasks: BackgroundTasks,
                         api_key: str = Security(get_api_key)):
    """Маршрут 1: Передаем ID модели с Hugging Face, и Python скачивает его в фоне"""
    repo_id = request.repo_id.strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id не может быть пустым")

    # Превращаем слэш в дефис для имени папки (например, desklib-ai-text-detector-v1.01)
    folder_name = repo_id.replace("/", "-")
    target_path = os.path.join(MODELS_DIR, folder_name)

    if os.path.exists(target_path) and repo_id not in download_tasks:
        return {"message": "Модель уже скачана ранее", "folder_name": folder_name}

    if repo_id in download_tasks and download_tasks[repo_id] == "Скачивание началось...":
        return {"message": "Модель уже находится в процессе загрузки", "status": download_tasks[repo_id]}

    # Запускаем асинхронное скачивание в фоне, чтобы API не зависало
    background_tasks.add_task(download_model_worker, repo_id, folder_name)

    return {"message": "Процесс загрузки запущен в фоновом режиме", "folder_name": folder_name}


@app.get("/api/models")
async def list_models(api_key: str = Security(get_api_key)):
    """Маршрут 2: Выводит список всех доступных локально моделей и текущих загрузок"""
    local_models = []
    if os.path.exists(MODELS_DIR):
        # Сканируем папку на наличие подпапок с моделями
        for name in os.listdir(MODELS_DIR):
            if os.path.isdir(os.path.join(MODELS_DIR, name)):
                local_models.append(name)

    return {
        "active_model": current_model_name if current_model_name else "Ни одна модель не активирована",
        "available_local_models": local_models,
        "active_download_statuses": download_tasks
    }


class ActivateRequest(BaseModel):
    folder_name: str  # Имя папки из списка доступных моделей


@app.post("/api/models/activate")
async def activate_model(request: ActivateRequest, api_key: str = Security(get_api_key)):
    """Дополнительный маршрут: Переключение активной модели в оперативной памяти"""
    success = load_model_into_memory(request.folder_name)
    if not success:
        raise HTTPException(status_code=400,
                            detail=f"Не удалось загрузить модель {request.folder_name}. Проверьте логи или статус загрузки.")
    return {"message": f"Модель {request.folder_name} успешно активирована!"}


class TextRequest(BaseModel):
    text: str


@app.post("/api/detect")
async def detect_text(request: TextRequest, api_key: str = Security(get_api_key)):
    """Основной маршрут детекции текста, работающий только на CPU"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=400,
            detail="На сервере не активирована ни одна модель. Сначала вызовите /api/models/activate"
        )

    clean_text = request.text.strip()
    if not clean_text:
        return {"confidence_score": 0.0, "is_ai_generated": False, "message": "Пустой текст"}

    try:
        # Токенизация текста
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Инференс без расчета градиентов
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Проверка, что модель вернула распределение как минимум для двух классов
        if probs.dim() == 0 or len(probs) < 2:
            raise HTTPException(status_code=500, detail="Модель вернула некорректное количество классов.")

        # Извлечение вероятностей по индексам
        human_prob = probs[0].item()
        ai_prob = probs[1].item()

        is_ai = ai_prob > human_prob
        final_score = ai_prob if is_ai else human_prob

        return {
            "is_ai_generated": is_ai,
            "confidence_score": round(final_score, 4),
            "probabilities": {
                "human": round(human_prob, 4),
                "ai": round(ai_prob, 4)
            },
            "active_model_used": current_model_name,
            "text_preview": clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")