import os
import gc
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError

app = FastAPI(title="Dynamic AI Text Detector API")

# Конфигурация путей и безопасности
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Глобальные переменные для активной модели
current_model_name = None
model = None
tokenizer = None
device = "cpu"

# Статусы фонового скачивания моделей
download_tasks = {}

# Маппинг классов для вердикта (измените индексы под вашу модель, если необходимо)
LABELS = {
    0: "Текст написан человеком",
    1: "Текст сгенерирован ИИ"
}


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
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    repo_id: str
    folder_name: str


class PredictRequest(BaseModel):
    text: str


@app.get("/models")
async def list_models():
    """Маршрут для просмотра всех локальных и скачиваемых моделей"""
    result = {}

    # 1. Сканируем локальную директорию на наличие уже скачанных моделей
    if os.path.exists(MODELS_DIR):
        for entry in os.scandir(MODELS_DIR):
            if entry.is_dir():
                folder_name = entry.name
                if folder_name == current_model_name:
                    result[folder_name] = "Активирована"
                else:
                    result[folder_name] = "Скачана (не активна)"

    # 2. Добавляем в список модели, которые сейчас скачиваются в фоне
    for repo_id, current_status in download_tasks.items():
        if current_status == "Скачивание началось...":
            result[repo_id] = "Скачивается..."

    return {"models": result}


@app.post("/download", status_code=status.HTTP_202_ACCEPTED)
async def download_model(payload: DownloadRequest, background_tasks: BackgroundTasks):
    """Маршрут для запуска скачивания модели"""
    repo_id = payload.repo_id
    folder_name = payload.folder_name

    # Проверка: не выполняется ли скачивание сейчас
    if download_tasks.get(repo_id) == "Скачивание началось...":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Модель уже скачивается"
        )

    # Валидация: проверка существования модели на Hugging Face до запуска фонового потока
    try:
        model_info(repo_id)
    except RepositoryNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Репозиторий '{repo_id}' не найден на Hugging Face"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ошибка проверки репозитория: {str(e)}"
        )

    background_tasks.add_task(download_model_worker, repo_id, folder_name)
    return {"status": "Скачивание началось в фоновом режиме", "repo_id": repo_id}


@app.get("/download/status/{repo_id:path}")
async def get_status(repo_id: str):
    """Маршрут для проверки статуса скачивания"""
    status_msg = download_tasks.get(repo_id, "Не найдено")
    return {"repo_id": repo_id, "status": status_msg}


@app.post("/activate")
async def activate_model(folder_name: str):
    """Маршрут для активации локальной модели"""
    success = load_model_into_memory(folder_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось загрузить модель. Проверьте имя папки."
        )
    return {"status": "Модель активирована", "current_model": current_model_name}


@app.post("/predict")
async def predict(payload: PredictRequest):
    """Маршрут для предсказания текста с вынесением булевого вердикта"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена"
        )

    try:
        inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        # Рассчитываем вероятности
        probs = torch.softmax(logits, dim=-1)
        probabilities_list = probs.tolist()

        # Находим индекс класса с максимальной вероятностью
        predicted_class_id = torch.argmax(probs, dim=-1).item()

        # Булевый вердикт: True если ИИ (класс 1), False если человек (класс 0)
        is_ai_generated = (predicted_class_id == 1)

        return {
            "model": current_model_name,
            "is_text_ai": is_ai_generated,  # Возвращает true или false
            "probabilities": probabilities_list,
            "logits": logits.tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка инференса: {str(e)}"
        )