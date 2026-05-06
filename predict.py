from classifier import make_prediction, CharacterFaceLabelLoader
from params import manga_name_list, DATASET_PATH, trial_no
from utils.dataloader import *
from model.models import SimCLRModel
from yolo_8 import *
from PIL import Image
import pandas as pd

test_indices = list(range(64))

model = SimCLRModel()
model.load_state_dict(torch.load('/content/simclr-1.pt', weights_only=True))
model.eval()

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

def compute_accuracy(dataloader, name, train_prop = 0.2, knn = 20):
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
    for k in range(1,knn):
        predictions = knn_classify(test_vectors, training_vectors, training_labels, k)

        accuracy = np.sum(np.where(predictions == test_labels, 1, 0)) / len(test_labels)
        accuracy_k[k] = accuracy
    
    return pd.DataFrame({'name':name} | accuracy_k)

if __name__ == '__main__':
    out_file_name = f'acc-{trial_no}.csv'
    if not os.path.exists(out_file_name):
        with open(out_file_name, 'w+') as f:
            f.close()

    for manga_idx in test_indices:
        name = manga_name_list[manga_idx]
        annotations = annotation_loader(DATASET_PATH, name)
        print(annotations['book'].keys())
        pages = annotations['book']['pages']
        print(pages)

        page_imgs = [retrieve_page(DATASET_PATH, name, page_idx) for page_idx in range(len(pages))]
        
        page_boxes = get_crop(page_imgs)

        all_cropped_chars = []
        for boxes, img in zip(page_boxes, page_imgs):
            for crop_info in page_boxes:
                positions = crop_info['position']
                character_crop = img.crop(positions)
                all_cropped_chars.append(character_crop)

        # Correct true labels
        true_labels = []

        assert len(all_cropped_chars) == len(true_labels)

        dataloader = CharacterFaceLabelLoader(DATASET_PATH, all_cropped_chars, true_labels,simple_transform)
        dataloader.set_encoded_imags(model)
        acc_df = compute_accuracy(dataloader, name)
        acc_df.to_csv(out_file_name , mode = 'a')


