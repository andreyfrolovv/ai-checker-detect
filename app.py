import os
import gc
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import model_info, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from tqdm.auto import tqdm
from transformers import DebertaV2ForSequenceClassification, AutoConfig, AutoTokenizer
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
# робочее состояние
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


def create_progress_tracker(repo_id: str):
    """Фабрика для создания класса tqdm, связанного с конкретным repo_id."""

    class TrackDownloadProgress(tqdm):
        def display(self, *args, **kwargs):
            super().display(*args, **kwargs)
            # Переводим байты в мегабайты
            downloaded_mb = self.n / (1024 * 1024)
            total_mb = self.total / (1024 * 1024) if self.total else 0

            if total_mb > 0:
                percent = (self.n / self.total) * 100
                status_str = f"Скачивание: {downloaded_mb:.1f}MB из {total_mb:.1f}MB ({percent:.1f}%)"
            else:
                status_str = f"Скачивание: {downloaded_mb:.1f}MB"

            # Записываем статус в глобальный словарь
            download_tasks[repo_id] = status_str

    return TrackDownloadProgress


def download_model_worker(repo_id: str, folder_name: str):
    """Фоновая функция для скачивания модели с отслеживанием прогресса."""
    target_path = os.path.join(MODELS_DIR, folder_name)
    try:
        download_tasks[repo_id] = "Подготовка к скачиванию..."

        # Динамически создаем класс трекера конкретно под этот репозиторий
        progress_tracker_class = create_progress_tracker(repo_id)

        # Ошибка исправлена: убран не поддерживаемый аргумент desc
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_path,
            tqdm_class=progress_tracker_class
        )

        download_tasks[repo_id] = "Успешно скачано"
    except Exception as e:
        download_tasks[repo_id] = f"Ошибка: {str(e)}"

# Использование Mean Pooling вместо [CLS]
# 1. Архитектура с Mean Pooling
class DesklibAIDetectionModel(nn.Module):
    def init(self, config):
        super().init()
        # Имя переменной соответствует маппингу
        self.deberta = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(outputs[0].size()).float()
        pooled = torch.sum(outputs[0] * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return self.classifier(pooled)



def load_model_into_memory(folder_name: str) -> bool:
    """Вспомогательная функция для переключения модели в ОЗУ (CPU)"""
    global model, tokenizer, current_model_name
    target_path = os.path.join(MODELS_DIR, folder_name)

    if not os.path.exists(target_path):
        return False

    try:
        # Безопасное освобождение памяти
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)

        # Загружаем оригинальный конфиг
        config = AutoConfig.from_pretrained(target_path, trust_remote_code=True)

        # Инициализируем НАШ кастомный класс вместо стандартного
        model = DesklibAIDetectionModel(config)

        # Ищем файл весов
        safetensors_path = os.path.join(target_path, "model.safetensors")
        bin_path = os.path.join(target_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError("Не найден файл весов модели")

        # Переименовываем ключи на лету под структуру нашего класса
        corrected_state_dict = {}
        for key, value in state_dict.items():
            # Наш класс использует префикс "deberta.", а в файле весов "model."
            if key.startswith("model.encoder.") or key.startswith("model.embeddings."):
                new_key = key.replace("model.", "deberta.", 1)
            elif key == "model.LayerNorm.weight":
                new_key = "deberta.LayerNorm.weight"
            elif key == "model.LayerNorm.bias":
                new_key = "deberta.LayerNorm.bias"
            else:
                new_key = key
            corrected_state_dict[new_key] = value

        # Загружаем веса в СТРОГОМ режиме (strict=True)
        # Если веса классификатора сядут идеально, ошибка не возникнет
        corrected_dict = {
            (k.replace("model.", "deberta.") if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(corrected_dict, strict=True)
        #model.load_state_dict(corrected_state_dict, strict=True)
        print("[Успех] Все оригинальные веса, включая классификатор, загружены!")

        model.to(device)
        model.eval()

        current_model_name = folder_name
        return True
    except Exception as e:
        print(f"Ошибка активации модели {folder_name}: {e}")
        model = None
        tokenizer = None
        current_model_name = None
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
        if "Скачивание" in current_status or "Подготовка" in current_status:
            result[repo_id] = current_status

    return {"models": result}


@app.post("/download", status_code=status.HTTP_202_ACCEPTED)
async def download_model(payload: DownloadRequest, background_tasks: BackgroundTasks):
    """Маршрут для запуска скачивания модели"""
    repo_id = payload.repo_id
    folder_name = payload.folder_name

    # Проверка: не выполняется ли скачивание сейчас
    current_status = download_tasks.get(repo_id, "")
    if "Скачивание" in current_status or "Подготовка" in current_status:
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
async def predict(data: PredictRequest):  # или как называется ваша Pydantic-модель
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    # 1. Токенизация текста
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 2. Получаем сырой логит
    logits = outputs.logits  # Имеет форму [[значение]]

    # === ИСПРАВЛЕНИЕ БАГА: Используем torch.sigmoid вместо torch.softmax ===
    probs = torch.sigmoid(logits)

    # Получаем численные значения для ответа API
    logits_list = logits.cpu().tolist()
    probs_list = probs.cpu().tolist()

    # Извлекаем саму вероятность (число от 0.0 до 1.0)
    prob_value = probs_list[0][0]

    # Определяем вердикт (например, если вероятность ИИ > 0.5)
    verdict = prob_value > 0.5

    return {
        "model": current_model_name,
        "verdict": verdict,
        "probabilities": probs_list,
        "logits": logits_list
    }