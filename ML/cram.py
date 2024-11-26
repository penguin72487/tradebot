def calculate_memory_usage(d_model, nhead, num_layers, seq_len, batch_size, dim_feedforward, input_dim=1, float_size=4):
    """
    計算 Transformer 模型大致顯存需求。
    顯存需求與模型的主要參數呈正比關係：
    - d_model: 每個序列位置的向量維度
    - nhead: 注意力機制的頭數
    - num_layers: Transformer 編碼器層數
    - seq_len: 序列長度
    - batch_size: 批量大小
    - dim_feedforward: 前饋網絡的隱藏層維度
    - input_dim: 模型的輸入維度（默認為 1）
    - float_size: 浮點數大小（以字節為單位，默認為 4 bytes 對應於 float32）
    """
    # 計算注意力機制的顯存需求
    attention_memory = num_layers * nhead * batch_size * seq_len * seq_len * d_model * float_size

    # 計算前饋神經網絡的顯存需求
    feedforward_memory = num_layers * batch_size * seq_len * dim_feedforward * float_size

    # 計算輸入和輸出的顯存需求
    input_memory = batch_size * seq_len * input_dim * float_size
    output_memory = batch_size * seq_len * d_model * float_size

    # 總顯存需求
    total_memory = attention_memory + feedforward_memory + input_memory + output_memory

    # 以 MB 為單位返回顯存需求
    return total_memory / (1024 ** 2)

# 示例使用
memory_usage = calculate_memory_usage(d_model=256, nhead=8, num_layers=8, seq_len=512, batch_size=1, dim_feedforward=1024)
print(f"預估顯存需求: {memory_usage:.2f} MB")
