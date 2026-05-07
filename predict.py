from classifier import make_prediction, CharacterFaceLabelLoader, EncodedFaceLabelLoader
from params import manga_name_list, DATASET_PATH, TARGET_SIZE
from utils.dataloader import *
from Model.models import SimCLRModel
from yolo_8 import *
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

# Choose trained model here
trial_no = 2

test_indices = [64]
to_tensor = transforms.ToTensor()

model = SimCLRModel()
model.load_state_dict(torch.load(f'simclr-{trial_no}.pt', weights_only=True, map_location=torch.device('cpu')))
model.eval()

# PREDICT_MODE = 'yolo'
PREDICT_MODE = 'dataset'
KNN = 16

from sklearn.neighbors import KNeighborsClassifier
def knn_classify(new_data, training_vectors, training_labels, k):
    """
    k-NN using scikit-learn
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(training_vectors, training_labels)
    predictions = model.predict(new_data)
    return predictions

from sklearn.utils import shuffle

def compute_accuracy(dataloader, name, train_prop = 0.2, knn = KNN, all_annotation_count = None):
    num_train = int(train_prop * len(dataloader.encoded_imgs))
    num_test = len(dataloader.encoded_imgs) - num_train

    all_vectors = dataloader.encoded_imgs[:]
    all_labels = dataloader.labels[:]

    shuffled_indices = np.random.permutation(len(all_vectors))
    shuffled_vectors = all_vectors[shuffled_indices]
    shuffled_labels = all_labels[shuffled_indices]

    training_vectors = shuffled_vectors[:num_train]
    training_labels = shuffled_labels[:num_train]

    test_vectors = shuffled_vectors[num_train:]
    test_labels = shuffled_labels[num_train:]

    accuracy_k = {}
    for k in range(1,knn+1):
        predictions = knn_classify(test_vectors, training_vectors, training_labels, k)

        if all_annotation_count is None:
            all_annotation_count = len(test_labels)
        accuracy = np.sum(np.where(predictions == test_labels, 1, 0)) / all_annotation_count
        accuracy_k[k] = accuracy
    
    k_max = np.argmax(list(accuracy_k.values())) + 1
    max_acc = accuracy_k[k_max]
    
    return pd.DataFrame({'name':name, 'k_max': k_max, 'max_acc': max_acc} | accuracy_k, index=[0])

if __name__ == '__main__':
    print("trial no ", trial_no)
    print("For ", PREDICT_MODE)
    out_file_name = f'acc-{trial_no}-{PREDICT_MODE}.csv'
    if not os.path.exists(out_file_name):
        with open(out_file_name, 'w+') as f:
            f.close()

    for manga_idx in test_indices:
        name = manga_name_list[manga_idx]
        
        all_cropped_chars = []
        true_labels = []

        if PREDICT_MODE == 'yolo':
            annotations = annotation_loader(DATASET_PATH, name)
            print(annotations['book'].keys())
            pages = annotations['book']['pages']['page']
            print(len(pages))

            annotation_boxes_count = np.sum([
                len(page.get('face')) for page in pages if 'face' in page
            ])

            char_id, _ = get_character_list(annotations)
            
            for page_num in range(len(pages)):
                page_img = retrieve_page(DATASET_PATH, name, page_num)
                page_boxes = get_crop_boxes_from_yolo(page_img)[0]
                all_positions = [crop_info['position'] for crop_info in page_boxes]
            
                results = get_paired_pred_boxes_labels(all_positions, annotations, page_num)
                all_cropped_chars.extend([
                    retrieve_page(DATASET_PATH, name, page_num, pair[0], target_size=TARGET_SIZE) for pair in results
                ])
                true_labels.extend([char_id.index(pair[1]) for pair in results])

            
            dataloader = CharacterFaceLabelLoader(all_cropped_chars, true_labels, to_tensor)
            dataloader.set_encoded_imags(model)
        
        if PREDICT_MODE == 'dataset':
            annotation_boxes_count = None
            print("[prog] Data Load")
            annotations = annotation_loader(DATASET_PATH, name)
            chars_dict = create_data(DATASET_PATH, annotations, target_size = TARGET_SIZE)
            map_idx_to_id = {}
            for i, (char_id, char_dict) in tqdm(enumerate(chars_dict.items())):
                map_idx_to_id[i] = char_id
                for img in char_dict['face_imgs']:
                    true_labels.append(i)
                    all_cropped_chars.append(img)

            dataloader = CharacterFaceLabelLoader(all_cropped_chars, true_labels, to_tensor)
            dataloader.set_encoded_imags(model)
        
        acc_df = compute_accuracy(dataloader, name, all_annotation_count = annotation_boxes_count)
        acc_df.to_csv(out_file_name , mode = 'a')
        print("save to ", out_file_name)


