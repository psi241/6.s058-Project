from dotenv import load_dotenv
from utils.dataloader import get_book_list
import os

# Load environment variables from .env file
load_dotenv()  
DATASET_PATH = os.getenv('DATASET_PATH')
manga_name_list = get_book_list(DATASET_PATH)

TARGET_SIZE = (112,112)
trained_manga_indices = list(range(32))

trial_no = 1

NUM_EPOCH_CONTRAST = 15