# run_step4_check.py
from src.paperpal.config import Config
from src.paperpal.data_prep import prepare_dataset

def main():
    cfg = Config()
    stats = prepare_dataset(cfg)
    print("✅ Step 4 complete:", stats)
    print("Outputs in:", cfg.processed_dir)

if __name__ == "__main__":
    main()

# yoyo