import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
import math
from collections import Counter

# Lớp Dataset để tạo bộ dữ liệu
class TextDataset(Dataset):
    def __init__(self, random_text_path, encrypted_text_path):
        self.data = []
        self.labels = []

        with open(random_text_path, 'r') as f:
            content1 = [line.strip() for line in f if line.strip()]

        with open(encrypted_text_path, 'r') as f2:
            content2 = [line.strip() for line in f2 if line.strip()]

        if len(content1) < 5000 or len(content2) < 5000:
            raise ValueError("Mỗi tệp phải chứa ít nhất 5000 mẫu.")

        print(f"Đã tải {len(content1)} văn bản ngẫu nhiên và {len(content2)} văn bản mã hóa.")

        for x in content1:
            text, entropy = x.rsplit(',', 1)
            self.data.append((text, float(entropy)))
            self.labels.append(0)

        for x in content2:
            text, entropy = x.rsplit(',', 1)
            self.data.append((text, float(entropy)))
            self.labels.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, entropy = self.data[idx]
        label = self.labels[idx]
        text_tensor = torch.tensor([(ord(c) - ord('A')) / 25 for c in text], dtype=torch.float32).unsqueeze(1)
        entropy_tensor = torch.tensor([entropy], dtype=torch.float32).unsqueeze(0)
        return text_tensor, entropy_tensor, label

# Hàm để tạo batch từ các mẫu có độ dài khác nhau
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    texts = [item[0] for item in batch]
    entropies = torch.cat([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.int64)
    padded_texts = pad_sequence(texts, batch_first=True)
    return padded_texts, entropies, labels, lengths

# Thiết lập mô hình học sâu
class RandomnessChecker(nn.Module):
    def __init__(self):
        super(RandomnessChecker, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128 + 1, 64)  # Thêm 1 cho giá trị entropy
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, entropies, lengths):
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        x = hn[-1]
        x = torch.cat((x, entropies), dim=1)  # Kết hợp giá trị entropy vào đầu ra của LSTM
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

# Hàm tính entropy của văn bản
def calculate_entropy(text):
    counter = Counter(text)
    total_chars = len(text)
    entropy = -sum((count / total_chars) * math.log2(count / total_chars) for count in counter.values())
    return entropy

# Cập nhật phần kiểm tra tính ngẫu nhiên từ văn bản người dùng
def check_randomness(text, model):
    entropy = calculate_entropy(text)
    text_tensor = torch.tensor([(ord(c) - ord('A')) / 25 for c in text], dtype=torch.float32).unsqueeze(1).unsqueeze(0)
    entropy_tensor = torch.tensor([entropy], dtype=torch.float32).unsqueeze(0)
    length = torch.tensor([len(text)], dtype=torch.int64)
    with torch.no_grad():
        output = model(text_tensor, entropy_tensor, length)
    return "Vigenère" if output.item() > 0.5 else "Ngẫu nhiên"

# Huấn luyện mô hình
def train_model(random_text_path, encrypted_text_path):
    dataset = TextDataset(random_text_path, encrypted_text_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = RandomnessChecker()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    patience = 5
    trigger_times = 0

    try:
        for epoch in range(10):
            print(f"Starting epoch {epoch + 1}")
            epoch_loss = 0
            for texts, entropies, labels, lengths in dataloader:
                optimizer.zero_grad()
                outputs = model(texts, entropies, lengths)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                trigger_times = 0
                torch.save(model.state_dict(), "randomness_checker_model.pth")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping triggered")
                    break
    except Exception as e:
        print(f"An error occurred during training: {e}")

    return model

def load_model():
    model = RandomnessChecker()
    model.load_state_dict(torch.load("randomness_checker_model.pth"), strict=False)
    return model

# Hàm chính
if __name__ == "__main__":
    random_text_path = 'ngau_nhien_with_entropy.txt'
    encrypted_text_path = 'vigenere_with_entropy.txt'

    # Xóa mô hình đã lưu nếu tồn tại
    if os.path.exists("randomness_checker_model.pth"):
        os.remove("randomness_checker_model.pth")

    model = train_model(random_text_path, encrypted_text_path)

    print("Mô hình Training đã hoàn thành. Bạn hãy kiểm tra tính ngẫu nhiên của văn bản.")

    while True:
        user_input = input("Nhập văn bản cần kiểm tra tính ngẫu nhiên hoặc ấn EXIT để thoát : ").upper()
        if user_input == 'EXIT':
            print('Kết thúc chương trình')
            break
        else:
            print(f"Văn bản nhập vào: {user_input}")
            print(f"Dự đoán: {check_randomness(user_input, model)}")
