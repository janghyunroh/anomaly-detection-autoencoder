import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, n_layers, dropout=0.2):
        """
        Args:
            input_dim (int): 입력 피처 개수 (X, Y, Z 축 가속도 데이터므로 3)
            hidden_dim (int): LSTM 히든 레이어의 차원
            seq_length (int): 입력 시퀀스 길이 (우리의 경우 30)
            n_layers (int): LSTM 레이어 개수
            dropout (float): 드롭아웃 비율
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.n_layers = n_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 최종 출력층
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # 입력 시퀀스 길이 확인
        if x.size(1) != self.seq_length:
            raise ValueError(f"Expected sequence length: {self.seq_length}, but got: {x.size(1)}")
            
        # Encoding
        _, (hidden, cell) = self.encoder(x)
        
        # 디코더의 입력 시퀀스 생성
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # Decoding
        decoder_output, _ = self.decoder(decoder_input)
        
        # 최종 출력
        output = self.output_layer(decoder_output)
        
        return output
    
    def get_reconstruction_error(self, x):
        """입력 데이터의 재구성 오차를 계산"""
        #x_reconst = self.forward(x)
        #relative_error = torch.abs(x - x_reconst) / (torch.abs(x) + 1e-6)
        #return torch.mean(relative_error, dim=(1, 2))
        
        x_reconst = self.forward(x)
        return torch.mean((x - x_reconst) ** 2, dim=(1, 2))