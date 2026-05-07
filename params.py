from dotenv import load_dotenv
from utils.dataloader import get_book_list
import os

# Load environment variables from .env file
load_dotenv()  
DATASET_PATH = '/Users/nathan/Library/Mobile Documents/com~apple~CloudDocs/1-MIT/Spring 2026/6.S058/6.s058-Project'
manga_name_list = get_book_list(DATASET_PATH)

TARGET_SIZE = (112,112)
trained_manga_indices = list(range(16))

trial_no = 3

NUM_EPOCH_CONTRAST = 10

temperature = 0.5
batch_size = 32
learning_rate = 5e-4
freeze_base = True