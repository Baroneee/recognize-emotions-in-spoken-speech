import torch
import torch.nn as nn
import torchtext.vocab as vocab_utils
from data import vocab, vocab_size

class RNNMODEL(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained=False):
        super().__init__()
        # Khởi tạo embedding layer (dùng GloVe nếu pretrained=True)
        # [Sinh viên bổ sung: dùng nn.Embedding, xứ lý pretrained với GloVe]
        if pretrained:
            # Tải GloVe 100d
            glove = vocab_utils.GloVe(name='6B', dim=embedding_dim)
            # Tạo embedding matrix với từ vựng có sẵn
            embedding_weights = torch.zeros(vocab_size, embedding_dim)
            for word, idx in vocab.items():
                if word in glove.stoi:
                    embedding_weights[idx] = glove.vectors[glove.stoi[word]]
                else:
                    embedding_weights[idx] = torch.randn(embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Khởi tạo Khối RNN Layer
        # [Sinh viên bổ sung: dùng nn.RNN với batch_first=True]
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Khởi tạo tầng Dense để dự đoán 3 nhãn
        # [Sinh viên bổ sung: dùng nn.Linear, nhận hidden state từ RNN]
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Chuyền text vào tầng embedding
        # [Sinh viên bổ sung]
        embedded = self.embedding(text)
        # Dưa qua Khối RNN để lấy hidden state cuối
        # [Sinh viên bổ sung]
        output, hidden = self.rnn(embedded)
        # Dưa hidden state qua tầng Dense để dự đoán 3 nhãn
        # [Sinh viên bổ sung]
        logits = self.fc(hidden.squeeze(0))
        return logits # [Sinh viên bổ sung: trả về kết quả dự đoán]