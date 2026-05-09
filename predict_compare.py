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
test_indices = list(range(0,109))
to_tensor = transforms.ToTensor()
PREDICT_MODE = 'dataset-compare'

KNN = 16
train_prop = 0.2

from sklearn.neighbors import KNeighborsClassifier
def knn_classify(new_data, training_vectors, training_labels, k):
    """
    k-NN using scikit-learn
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(training_vectors, training_labels)
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    models = {}
    for trial_no in [2,3,4]:
        model = SimCLRModel()
        model.load_state_dict(torch.load(f'simclr-{trial_no}.pt', weights_only=True, map_location=torch.device('cpu')))
        model.eval()
        models[trial_no] = model

        out_file_name = f'acc-{trial_no}-{PREDICT_MODE}.csv'
        if not os.path.exists(out_file_name):
            with open(out_file_name, 'w+') as f:
                f.close()

    for manga_idx in test_indices:
        name = manga_name_list[manga_idx]
        all_cropped_chars = []
        true_labels = []

        print("[prog] Data Load")
        annotations = annotation_loader(DATASET_PATH, name)
        chars_dict = create_data(DATASET_PATH, annotations, target_size = TARGET_SIZE)

        map_idx_to_id = {}
        for i, (char_id, char_dict) in tqdm(enumerate(chars_dict.items())):
            map_idx_to_id[i] = char_id
            for img in char_dict['face_imgs']:
                true_labels.append(i)
                all_cropped_chars.append(img)

        # Seperate Test and Train set  
        dataloader = CharacterFaceLabelLoader(all_cropped_chars, true_labels, to_tensor)  
        num_train = int(train_prop * len(dataloader.labels))

        shuffled_indices = np.random.permutation(len(dataloader.labels[:]))
        shuffled_labels = dataloader.labels[shuffled_indices]
        training_labels = shuffled_labels[:num_train]
        test_labels = shuffled_labels[num_train:]

        for trial_no in [2,3,4]:
            model = models[trial_no]
            out_file_name = f'acc-{trial_no}-{PREDICT_MODE}.csv'
            dataloader.set_encoded_imags(model)

            shuffled_vectors = dataloader.encoded_imgs[shuffled_indices]
            training_vectors = shuffled_vectors[:num_train]
            test_vectors = shuffled_vectors[num_train:]
            
            accuracy_k = {}
            for k in range(1,KNN+1):
                predictions = knn_classify(test_vectors, training_vectors, training_labels, k)

                accuracy = np.sum(np.where(predictions == test_labels, 1, 0))
                accuracy_k[k] = accuracy
            
            k_max = np.argmax(list(accuracy_k.values())) + 1
            max_acc = accuracy_k[k_max]
            
            acc_df = pd.DataFrame({'name':name, 'k_max': k_max, 'max_acc': max_acc, 'num_tested':  len(test_vectors), 'total':  len(test_labels)} | accuracy_k, index=[manga_idx])
            acc_df.to_csv(out_file_name , mode = 'a', header = False)
            print("save to ", out_file_name)


