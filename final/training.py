import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Replace with the path to your Marathi dataset
marathi_data_path = "balanced_dataset.xlsx"

# Load and preprocess your dataset
# Assuming your dataset has columns 'text' and 'label' (0 or 1)
# You might need to adapt this based on your actual data format
def load_marathi_dataset(file_path):
    try:
        # Load your dataset using pandas or any other method
        df = pd.read_excel(file_path)

        # Assuming your dataset has columns 'text' and 'label'
        texts = df['Word'].tolist()
        labels = df['Label'].tolist()

        return texts, labels

    except KeyError as e:
        print(f"Error: Column {e} not found in the dataset. Check your column names.")
        # You might want to handle this error or raise it for further debugging.

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other exceptions if needed

    return None, None

class MarathiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

# Load and preprocess the dataset
texts, labels = load_marathi_dataset(marathi_data_path)

# Split the dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

# Load the BERT tokenizer and model with force_download and resume_download options
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', force_download=True, resume_download=False)
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2, force_download=True, resume_download=False)

# Create datasets and data loaders
train_dataset = MarathiDataset(train_texts, train_labels, tokenizer)
val_dataset = MarathiDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Set up training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            correct_preds += torch.sum(predictions == labels).item()
            total_samples += len(labels)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_preds / total_samples

        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

# Save the fine-tuned model
model.save_pretrained('/content/drive/MyDrive/dataset/model.h5')
tokenizer.save_pretrained('/content/drive/MyDrive/dataset/tokenizer.json')
