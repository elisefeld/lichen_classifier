from utils import scraping
from config import Config
cfg = Config()

scraping.train_test_split(source_dir=cfg.full_img_dir,
                          dest_train_dir=cfg.train_dir,
                          dest_test_dir=cfg.test_dir,
                          dest_val_dir=cfg.val_dir,
                          ratio=cfg.val_test_split)