from dotenv import load_dotenv
from utils.dataloader import get_book_list
import os
import numpy as np

# Load environment variables from .env file
load_dotenv()  
DATASET_PATH = os.getenv('DATASET_PATH')
manga_name_list = get_book_list(DATASET_PATH)

TARGET_SIZE = (112,112)
trained_manga_indices = np.array([82, 38, 29, 11, 95, 70, 28, 60, 40, 79, 45, 75, 14, 68, 33, 92, 51, 85, 8, 47, 55, 84, 61, 59, 16, 91, 15, 83, 86, 26, 1, 67])

trial_no = 5

NUM_EPOCH_CONTRAST = 10

temperature = 0.5
batch_size = 32
learning_rate = 5e-4
freeze_base = True
