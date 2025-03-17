import pandas as pd
import sqlalchemy
import joblib

# **✅ 连接数据库**
DATABASE_URL = "mssql+pyodbc://sa:123456@DESKTOP-AR78FVS\\MSSQLSERVER01/EnglishLearningDB?driver=ODBC+Driver+17+for+SQL+Server"
engine = sqlalchemy.create_engine(DATABASE_URL)

# **✅ 直接查询 `PersonalWords`，不需要 `JOIN ClickLogs`**
query = """
    SELECT UserId, Word, Familiarity, CreatedAt
    FROM PersonalWords
    ORDER BY UserId, CreatedAt;
"""
df = pd.read_sql(query, engine)

print(f"✅ 加载了 {len(df)} 个单词")

# **✅ 训练 `word_encoder`**
from sklearn.preprocessing import LabelEncoder
word_encoder = LabelEncoder()
df["WordId"] = word_encoder.fit_transform(df["Word"])

# **✅ 保存 `word_encoder.pkl`**
joblib.dump(word_encoder, "word_encoder.pkl")
print("✅ word_encoder 已保存！")


