import torch
import joblib
import train_transformer
import pandas as pd
import data_loader

# **✅ 加载模型**
model = train_transformer.FamiliarityTransformer(vocab_size=len(train_transformer.word_encoder.classes_) + 1)
model.load_state_dict(torch.load("familiarity_transformer.pth"))
model.eval()

# **✅ 加载 word_encoder**
word_encoder = joblib.load("word_encoder.pkl")
VOCAB_SIZE = len(word_encoder.classes_)

# **✅ 获取“熟悉度最低”的单词**
def get_least_familiar_words(num_predictions=5):
    query = f"""
        SELECT TOP {num_predictions} Word
        FROM PersonalWords
        ORDER BY Familiarity ASC, NEWID();
    """
    df = pd.read_sql(query, data_loader.engine)
    return df["Word"].tolist()

# **✅ 结合 Transformer 预测和 熟悉度最低单词**
def predict_next_words(past_words, num_predictions=5):
    # **Transformer 预测**
    encoded_words = [word_encoder.transform([word])[0] + 1 for word in past_words]
    input_tensor = torch.tensor(encoded_words, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        num_predictions = min(num_predictions, VOCAB_SIZE)
        top_indices = torch.topk(logits, num_predictions, dim=-1).indices.squeeze().tolist()

    predicted_words = word_encoder.inverse_transform(top_indices)

    # **获取熟悉度最低的单词**
    least_familiar_words = get_least_familiar_words(num_predictions)

    return {
        "transformer_predictions": predicted_words,
        "least_familiar_words": least_familiar_words
    }

# **✅ 测试：同时获取 Transformer 预测 和 熟悉度最低的单词**
print(f"Recommended words: {predict_next_words(['artificial', 'banana'], num_predictions=5)}")




