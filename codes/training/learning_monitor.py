import matplotlib.pyplot as plt
import tensorflow as tf
import os

class ReconstructionErrorCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_dir='/home/cnserver/projects/AI/results/graphs/'):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        # logs에서 훈련 및 검증 손실 가져오기
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        # 손실 기록
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    def on_train_end(self, logs=None):
        # 디렉토리가 없으면 생성
        os.makedirs(self.output_dir, exist_ok=True)

        # 학습 종료 후 그래프 작성 및 저장
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Reconstruction Error')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Reconstruction Error')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Error')
        plt.title('Reconstruction Error by Epoch')
        plt.legend()
        plt.grid(True)

        # output_dir 경로에 그래프 파일 저장
        file_path = os.path.join(self.output_dir, 'reconstruction_error_by_epoch.png')
        plt.savefig(file_path)

        plt.show()

