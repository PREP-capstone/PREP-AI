import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizerFast, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using High-Performance Device for Large Model: {device}")

# 2. 데이터셋 클래스 정의
class HealthcareDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['combined_text'].values
        self.labels = df['category_id'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_large_model():
    # 3. 데이터 로드 및 전처리
    df = pd.read_csv('pilot_all_labeled_completed.csv')
    
    # EXC(예외) 데이터 제외
    df = df[df['category_id'] != 'EXC'].copy()
    df['category_id'] = df['category_id'].astype(int)

    # 텍스트 + 수집 데이터 결합
    df['collected_data'] = df['collected_data'].fillna('')
    df['combined_text'] = df['description'] + " [SEP] 수집 데이터: " + df['collected_data']

    # Stratified Split (클래스 비율 유지 분할)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category_id'])

    # 4. 소수 클래스 불균형 해소를 위한 가중치 계산 (Weighted Loss)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['category_id']),
        y=train_df['category_id'].values
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class Weights Applied: {class_weights}")

    # 5. 모델 변경: klue/roberta-base -> klue/roberta-large (체급 업그레이드)
    model_name = "klue/roberta-large"
    print(f"Loading pretrained model: {model_name} ...")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    
    num_labels = df['category_id'].nunique()
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    train_dataset = HealthcareDataset(train_df, tokenizer)
    val_dataset = HealthcareDataset(val_df, tokenizer)

    # 라지 모델의 VRAM 효율성을 위해 배치 사이즈 16으로 설정 (필요시 8로 조절 가능)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 6. 하이퍼파라미터 설정 (Epochs 20, Weight Decay 0.05)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
    epochs = 20
    total_steps = len(train_loader) * epochs
    
    # Warmup ratio 10% 적용
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )

    # 가중치 손실 함수 설정
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_macro_f1 = 0.0

    print("--- Starting RoBERTa-Large Advanced Training ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # 검증 평가
        model.eval()
        val_preds, val_trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        acc = accuracy_score(val_trues, val_preds)
        macro_f1 = f1_score(val_trues, val_preds, average='macro')

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} | Val Macro F1: {macro_f1:.4f}")

        # 최고 성능 모델 갱신 저장
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            model.save_pretrained('./best_healthcare_model_large')
            tokenizer.save_pretrained('./best_healthcare_model_large')
            print(f"✨ [New Best] Large Model saved with Macro F1: {best_macro_f1:.4f}")

if __name__ == "__main__":
    train_large_model()