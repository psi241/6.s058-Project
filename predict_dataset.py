from classifier import make_prediction, CharacterFaceLabelLoader, EncodedFaceLabelLoader
from params import manga_name_list, DATASET_PATH, TARGET_SIZE
from utils.dataloader import *
from Model.models import SimCLRModel
from yolo_8 import *
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from pprint import pprint

# Choose trained model here
trial_no = 3

test_indices = list(range(0, 109))
to_tensor = transforms.ToTensor()

model = SimCLRModel()
model.load_state_dict(torch.load(f'simclr-{trial_no}.pt', weights_only=True, map_location=torch.device('cpu')))
model.eval()

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


def compute_accuracy(test, train, filename, mode):
    accuracy_k = {}
    for k in range(1,KNN+1):
        predictions = knn_classify(
            test.encoded_imgs,
            train.encoded_imgs, 
            train.labels,
            k
        )
        
        accuracy_k[k] = np.sum(np.where(predictions == test.labels, 1, 0))

    k_max = np.argmax(list(accuracy_k.values())) + 1
    max_acc = accuracy_k[k_max]
    
    row = pd.DataFrame({'name':name, 'k_max': k_max, 'max_acc': max_acc, 'all_data': len(test.labels)} | accuracy_k, index=[0])
    row.to_csv(filename, mode = 'a', header = False)
    
    print(f"save to {filename} highest accuracy for {mode} : {max_acc/len(test.labels):.5f}")

if __name__ == '__main__':
    print("trial no ", trial_no)
    # Output
    yolo_filename = f'acc-{trial_no}-yolo-charbase.csv'
    if not os.path.exists(yolo_filename ):
        with open(yolo_filename , 'w+') as f:
            f.close()
        header = pd.DataFrame(columns=['name', 'k_max', 'max_acc'] + list(range(1, KNN+1)))
        header.to_csv(yolo_filename )

    ds_filename = f'acc-{trial_no}-ds-charbase.csv'
    if not os.path.exists(ds_filename):
        with open(ds_filename, 'w+') as f:
            f.close()
        header = pd.DataFrame(columns=['name', 'k_max', 'max_acc'] + list(range(1, KNN+1)))
        header.to_csv(ds_filename)

    for manga_idx in test_indices:
        name = manga_name_list[manga_idx]
        annotations = annotation_loader(DATASET_PATH, name)
        print(f"Compile name: {name}")

        # Create training data from dataset
        chars_dict = create_data(DATASET_PATH, annotations, target_size = TARGET_SIZE)
        dict_id_to_list = {v['id'] : v['face_imgs'] for v in chars_dict.values()}

        print("Total char counts = ", np.sum([len(v) for v in dict_id_to_list.values()]))

        train_data, train_label, test_data, test_label = train_test_split_data(dict_id_to_list, train_ratio=0.2, random_seed=20)

        train_dataloader = CharacterFaceLabelLoader(train_data, train_label, to_tensor, label_tensor = False)
        train_dataloader.set_encoded_imags(model)

        validate_dataloader = CharacterFaceLabelLoader(test_data, test_label, to_tensor, label_tensor = False)
        validate_dataloader.set_encoded_imags(model)

        # Retrieve image from prediction
        pages = annotations['book']['pages']['page']

        annotation_boxes_count = np.sum([
            len(page.get('face')) for page in pages if 'face' in page
        ])

        char_id, _ = get_character_list(annotations)
        
        all_cropped_chars = []
        true_labels = []

        for page_num in range(len(pages)):
            page_img = retrieve_page(DATASET_PATH, name, page_num)
            page_boxes = get_crop_boxes_from_yolo(page_img)[0]

            all_positions = [crop_info['position'] for crop_info in page_boxes]
        
            results = get_paired_pred_boxes_labels(all_positions, annotations, page_num)
            all_cropped_chars.extend([
                retrieve_page(DATASET_PATH, name, page_num, pair[0], target_size=TARGET_SIZE) for pair in results
            ])

            true_labels.extend([pair[1] for pair in results])

        test_dataloader = CharacterFaceLabelLoader(all_cropped_chars, true_labels, to_tensor, label_tensor = False)
        test_dataloader.set_encoded_imags(model)

        print("Compute accuracy")
        compute_accuracy(test_dataloader, train_dataloader, yolo_filename, "yolo")
        compute_accuracy(validate_dataloader, train_dataloader, ds_filename, "ds")
