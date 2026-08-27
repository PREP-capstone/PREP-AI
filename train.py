import os
from rich import print
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW

# 1. 디바이스 설정 (GPU 가속 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 축2(function_type) 라벨 매핑: A/B/C/D -> 0/1/2/3
FUNCTION_TYPE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
NUM_FUNCTION_LABELS = len(FUNCTION_TYPE_MAP)

# category_id(EXC 제외 시) 클래스 개수는 데이터에서 동적으로 계산
CATEGORY_IGNORE_INDEX = -100


# 2. 데이터셋 클래스 정의 (텍스트 + 수집 데이터 결합 전처리, 축1/축2 동시 라벨링)
class HealthcareDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['combined_text'].values
        self.category_labels = df['category_label'].values      # 축1: 0~N-1, EXC는 -100
        self.function_labels = df['function_label'].values      # 축2: 0~3
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category_labels': torch.tensor(int(self.category_labels[idx]), dtype=torch.long),
            'function_labels': torch.tensor(int(self.function_labels[idx]), dtype=torch.long),
        }


# 3. 멀티헤드 모델 정의: 공유 인코더 + 축1(category) 헤드 + 축2(function_type) 헤드
class MultiHeadHealthcareModel(nn.Module):
    def __init__(self, model_name, num_category_labels, num_function_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(hidden_size, num_category_labels)
        self.function_head = nn.Linear(hidden_size, num_function_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 임베딩
        pooled = self.dropout(cls_output)

        category_logits = self.category_head(pooled)
        function_logits = self.function_head(pooled)
        return category_logits, function_logits


def train_model():
    # 4. 데이터 로드 및 전처리
    df = pd.read_csv('pilot_all_labeled_completed.csv')

    # 축1(category_id): EXC는 라벨 자체는 유지하되 손실 계산에서 무시(-100) 처리
    category_classes = sorted(int(c) for c in df['category_id'].unique() if c != 'EXC')
    category_to_label = {c: i for i, c in enumerate(category_classes)}
    num_category_labels = len(category_classes)

    def map_category(value):
        if value == 'EXC':
            return CATEGORY_IGNORE_INDEX
        return category_to_label[int(value)]

    df['category_label'] = df['category_id'].apply(map_category)

    # 축2(function_type): 모든 행에 값이 있으므로 EXC 여부와 무관하게 전부 사용
    df['function_label'] = df['function_type'].map(FUNCTION_TYPE_MAP)

    # 설계서 요구사항: 서비스 설명(description) + 수집할 데이터(collected_data) 결합
    df['collected_data'] = df['collected_data'].fillna('')
    df['combined_text'] = df['description'] + " [SEP] 수집 데이터: " + df['collected_data']

    # Train / Validation 분할 (8:2) - category_id(EXC 포함) 기준으로 층화
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['category_id']
    )

    # 5. 모델 및 토크나이저 로드 (KLUE-RoBERTa-base)
    model_name = "klue/roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    model = MultiHeadHealthcareModel(model_name, num_category_labels, NUM_FUNCTION_LABELS)
    model.to(device)

    # DataLoader 생성
    train_dataset = HealthcareDataset(train_df, tokenizer)
    val_dataset = HealthcareDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 6. 학습 설정 (설계서 반영: AdamW, LR 2e-5, Early Stopping 등)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 10
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    category_criterion = nn.CrossEntropyLoss(ignore_index=CATEGORY_IGNORE_INDEX)
    function_criterion = nn.CrossEntropyLoss()

    best_avg_macro_f1 = 0.0

    # 7. 학습 루프 (Training Loop) - 두 헤드의 손실을 합산하여 역전파
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            category_labels = batch['category_labels'].to(device)
            function_labels = batch['function_labels'].to(device)

            category_logits, function_logits = model(input_ids, attention_mask)

            loss_category = category_criterion(category_logits, category_labels)
            loss_function = function_criterion(function_logits, function_labels)
            loss = loss_category + loss_function
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()
            scheduler.step()

        # 검증(Validation) - 축1/축2 각각 Macro F1 계산 (축1은 EXC=-100 제외)
        model.eval()
        cat_preds, cat_trues = [], []
        func_preds, func_trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                category_labels = batch['category_labels'].to(device)
                function_labels = batch['function_labels'].to(device)

                category_logits, function_logits = model(input_ids, attention_mask)
                cat_pred = torch.argmax(category_logits, dim=1)
                func_pred = torch.argmax(function_logits, dim=1)

                valid_mask = category_labels != CATEGORY_IGNORE_INDEX
                cat_preds.extend(cat_pred[valid_mask].cpu().numpy())
                cat_trues.extend(category_labels[valid_mask].cpu().numpy())

                func_preds.extend(func_pred.cpu().numpy())
                func_trues.extend(function_labels.cpu().numpy())

        cat_acc = accuracy_score(cat_trues, cat_preds)
        cat_macro_f1 = f1_score(cat_trues, cat_preds, average='macro')
        func_acc = accuracy_score(func_trues, func_preds)
        func_macro_f1 = f1_score(func_trues, func_preds, average='macro')
        avg_macro_f1 = (cat_macro_f1 + func_macro_f1) / 2

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} "
            f"| [축1] Acc: {cat_acc:.4f} F1: {cat_macro_f1:.4f} "
            f"| [축2] Acc: {func_acc:.4f} F1: {func_macro_f1:.4f} "
            f"| Avg F1: {avg_macro_f1:.4f}"
        )

        # 최고 성능 모델 저장 (Checkpoint) - 두 축의 평균 Macro F1 기준
        if avg_macro_f1 > best_avg_macro_f1:
            best_avg_macro_f1 = avg_macro_f1
            save_dir = './best_healthcare_model_2line'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
            tokenizer.save_pretrained(save_dir)
            torch.save(
                {
                    'category_classes': category_classes,       # label index -> 원본 category_id
                    'function_type_map': FUNCTION_TYPE_MAP,      # 'A'/'B'/'C'/'D' -> label index
                    'model_name': model_name,
                    'num_category_labels': num_category_labels,
                    'num_function_labels': NUM_FUNCTION_LABELS,
                },
                os.path.join(save_dir, 'label_config.pt'),
            )
            print(f"-> Best model saved with Avg Macro F1: {best_avg_macro_f1:.4f} (축1: {cat_macro_f1:.4f}, 축2: {func_macro_f1:.4f})")


if __name__ == "__main__":
    train_model()
