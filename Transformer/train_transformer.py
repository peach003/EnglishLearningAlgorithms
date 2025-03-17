import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import data_loader  #  直接导入 data_loader
import joblib

# **1. 加载数据**
df = data_loader.df  # **直接从 data_loader.py 获取 df**
word_encoder = joblib.load("word_encoder.pkl")  # **加载 `word_encoder.pkl`**
VOCAB_SIZE = len(word_encoder.classes_) + 1  # **+1 避免索引超界**

# **2. 自定义 PyTorch 数据集**
class FamiliarityDataset(Dataset):
    def __init__(self, df):
        self.data = []
        grouped = df.groupby("UserId")
        for _, group in grouped:
            words = list(group["WordId"] + 1)  # **确保索引从 1 开始**
            familiarity = list(group["Familiarity"])

            for i in range(1, len(words)):
                input_seq = torch.tensor(words[:i], dtype=torch.long)
                target_familiarity = torch.tensor(familiarity[i], dtype=torch.float)  
                self.data.append((input_seq, target_familiarity))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# **3. 变长序列填充**
def collate_fn(batch):
    input_seqs, targets = zip(*batch)
    
    max_len = max(len(seq) for seq in input_seqs)
    padded_inputs = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in input_seqs]
    
    return torch.stack(padded_inputs), torch.tensor(targets, dtype=torch.float)

# ** 4. 创建数据加载器**
dataset = FamiliarityDataset(df)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# **5. 定义 Transformer 模型**
class FamiliarityTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, num_heads=2, hidden_dim=32):
        super(FamiliarityTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        # ** 输出 `VOCAB_SIZE` 个可能的单词（不再是 1 维度）**
        self.fc = nn.Linear(embed_dim, vocab_size)  

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])  # ** 预测整个词汇表的可能性**
    
# **6. 训练模型**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FamiliarityTransformer(vocab_size=VOCAB_SIZE).to(device)
criterion = nn.CrossEntropyLoss()  # **损失函数适用于多类别预测**
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for words, target in dataloader:
        words, target = words.to(device), target.to(device, dtype=torch.long)  # **转换为 `long` 类型**
        optimizer.zero_grad()
        output = model(words)  # ** 预测的是整个 `VOCAB_SIZE` 维度**
        
        loss = criterion(output, target)  # **使用 `CrossEntropyLoss()`**
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# ** 7. 保存模型**
joblib.dump(word_encoder, "word_encoder.pkl")
torch.save(model.state_dict(), "familiarity_transformer.pth")
print(" 训练完成，已保存模型和词汇表！")





