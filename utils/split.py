import json
import random

input_file = "data/all_train.jsonl"
train_output = "data/train.jsonl"
val_output = "data/val.jsonl"
split_ratio = 0.8

data = []

# 데이터 로드
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# 랜덤 셔플 (재현성을 위해 시드 고정)
random.seed(42)
random.shuffle(data)

# 8:2 분할 지점 계산
split_index = int(len(data) * split_ratio)

train_data = data[:split_index]
val_data = data[split_index:]


# 파일 저장 함수
def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


save_jsonl(train_data, train_output)
save_jsonl(val_data, val_output)

print(f"Total: {len(data)}")
print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
