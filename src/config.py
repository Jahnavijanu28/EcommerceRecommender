import torch
from pathlib import Path

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = DATA_DIR / 'models'
    OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
    
    # Model parameters
    EMBEDDING_DIM = 64
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Data
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    @staticmethod
    def get_device():
        """Get the device (cuda or cpu)"""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def setup_directories(cls):
        """Create all necessary directories"""
        for directory in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                         cls.MODELS_DIR, cls.OUTPUTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        print("✅ Directories created")

if __name__ == "__main__":
    Config.setup_directories()
    device = Config.get_device()
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
