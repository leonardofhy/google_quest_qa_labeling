from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

# 設定模型名稱與儲存路徑
model_name = "microsoft/deberta-v3-large"
save_path = "./deberta-v3-large"  # 這會在當前目錄建立一個資料夾

# 建立目錄
os.makedirs(save_path, exist_ok=True)

# 下載並儲存 Tokenizer, Config 和 Model
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
config.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Model saved to {save_path}")