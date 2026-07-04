import os
import shutil
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoTokenizer, AutoConfig
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel


# --- Исходный код модели ---
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Инициализация базовой модели
        self.model = AutoModel.from_config(config)
        # Определение классификатора
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Инициализация весов
        self.init_weights()

        # --- ДОБАВЬТЕ ЭТУ СТРОКУ ДЛЯ СОВМЕСТИМОСТИ С TRANSFORMERS 5.x ---
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Оставляем ваш forward без изменений...
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


def predict_single_text(text, model, tokenizer, device, threshold=0.5):
    encoded = tokenizer(
        text, padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    label = 1 if probability >= threshold else 0
    return probability, label


# --- FastAPI и Логика Управления Моделями ---
app = FastAPI(title="AI Text Detection API")

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)


class ModelManager:
    def __init__(self):
        self.active_model = None
        self.active_tokenizer = None
        self.active_model_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Статусы скачивания: "downloading", "completed", "failed"
        self.download_statuses = {}


manager = ModelManager()


# Схемы запросов Pydantic
class DownloadRequest(BaseModel):
    model_name: str  # Например: "desklib/ai-text-detector-v1.0"


class ActivateRequest(BaseModel):
    model_name: str


class CheckTextRequest(BaseModel):
    text: str
    threshold: float = 0.5
    max_len: int = 768


# Фоновая задача для скачивания
def download_model_task(model_name: str):
    try:
        manager.download_statuses[model_name] = "downloading"
        # Санитизация имени папки (замена / на _)
        folder_name = model_name.replace("/", "_")
        target_dir = os.path.join(MODELS_DIR, folder_name)

        # Скачивание конфига, токенайзера и весов через Hugging Face Hub
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = DesklibAIDetectionModel.from_pretrained(model_name, config=config)

        # Сохранение на диск
        config.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)
        model.save_pretrained(target_dir)

        manager.download_statuses[model_name] = "completed"
    except Exception as e:
        manager.download_statuses[model_name] = f"failed: {str(e)}"


# 1. Скачивание модели
@app.post("/models/download")
def download_model(payload: DownloadRequest, background_tasks: BackgroundTasks):
    model_name = payload.model_name
    if model_name in manager.download_statuses and manager.download_statuses[model_name] == "downloading":
        return {"message": "Модель уже скачивается", "status": "downloading"}

    background_tasks.add_task(download_model_task, model_name)
    return {"message": "Скачивание модели запущено в фоне", "model_name": model_name}


# 2. Список всех скачанных моделей и статусы
@app.get("/models")
def list_models():
    downloaded_folders = os.listdir(MODELS_DIR)
    return {
        "downloaded_models_folders": downloaded_folders,
        "download_tasks_statuses": manager.download_statuses,
        "active_model": manager.active_model_name
    }


# 3. Активация модели для использования
@app.post("/models/activate")
def activate_model(payload: ActivateRequest):
    folder_name = payload.model_name.replace("/", "_")
    model_path = os.path.join(MODELS_DIR, folder_name)

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Модель не найдена локально. Сначала скачайте её через /models/download"
        )

    try:
        # Очистка памяти перед загрузкой новой модели
        manager.active_model = None
        manager.active_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Загрузка локальных файлов
        config = AutoConfig.from_pretrained(model_path)
        manager.active_tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = DesklibAIDetectionModel.from_pretrained(model_path, config=config)
        manager.active_model = model.to(manager.device)
        manager.active_model_name = payload.model_name

        return {"message": f"Модель {payload.model_name} успешно активирована на {manager.device}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка активации: {str(e)}")


# 4. Отправка запроса проверки текста
@app.post("/predict")
def check_text(payload: CheckTextRequest):
    if not manager.active_model or not manager.active_tokenizer:
        raise HTTPException(
            status_code=400,
            detail="Нет активной модели. Сначала активируйте модель через /models/activate"
        )

    try:
        probability, label = predict_single_text(
            text=payload.text,
            model=manager.active_model,
            tokenizer=manager.active_tokenizer,
            device=manager.device,
            threshold=payload.threshold
        )

        return {
            "text": payload.text,
            "ai_probability": probability,
            "is_ai": bool(label),
            "applied_threshold": payload.threshold,
            "model_used": manager.active_model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа текста: {str(e)}")