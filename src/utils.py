import logging
import os
import tensorflow as tf
from tqdm import tqdm
import cv2

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from glob import glob
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score

# from keras import backend as K


def create_logger(model_name):
    os.makedirs(f'logs/{model_name}', exist_ok=True)
    logging.basicConfig(filename=f'logs/{model_name}/app.log', \
                    filemode='a', format='%(asctime)s -%(levelname)s- %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def gpu_managemenent():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print('memory growth set true')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def find_feature_vector(feature_extaction_model, input_images, embedding_size, BATCH_SIZE=256):    
    num_images             = len(input_images)
    input_images           = np.stack(np.array(input_images), axis=0)
    
    # custom batched prediction loop to avoid memory leak issues for now in the model.predict call
    feature_vector = np.empty([num_images, embedding_size], dtype=np.float32)  # pre-allocate required memory for array for efficiency

    BATCH_INDICES = np.arange(start=0, stop=num_images, step=BATCH_SIZE)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, num_images)  # add final batch_end row

    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end   = BATCH_INDICES[index + 1]  # last row of the batch
        feature_vector[batch_start:batch_end] = feature_extaction_model.predict_on_batch(input_images[batch_start:batch_end])
        
#     feature_vector         = feature_extaction_model.predict(input_images, batch_size = 256)
#     print(num_images, feature_vector.shape)
    feature_vector         = feature_vector.reshape((num_images, -1))
    # K.clear_session()
    
    return feature_vector

def cos_similarity(img, img_to_compare, num_closest):

    similarity    = cosine_similarity(img_to_compare, img) # compute cosine similarities between images
    similarity_pd = pd.DataFrame(similarity, columns=[0], index=range(len(img_to_compare)))

    sim           = similarity_pd[0].sort_values(ascending=False)[0:num_closest].index
    sim_score     = similarity_pd[0].sort_values(ascending=False)[0:num_closest].to_list()

    return np.array(sim), sim_score

def compute_metrics(y_truth, y_pred):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(y_truth, y_pred)
    
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_truth, y_pred)
    
    return fpr, tpr, thresholds, auc

    
def find_top_k_retrival(user_img, list_of_shop_images, fv_bag, max_k = 50):

    all_shop_product  = []
    
    anchor_feature_vector      = [fv_bag[str(user_img)]]
    all_shop_feature_vector    = []          # feature vector of user image


    # Iterate over all the remaining shop image
    for shop_img_path in list_of_shop_images:

        all_shop_feature_vector.append(fv_bag[str(shop_img_path)]) # feature vector of negative images
        all_shop_product.append(shop_img_path.split('/')[-3].split('_')[0])
        
    max_k = min(max_k, len(all_shop_product))
    cos_indices, sim_score = cos_similarity(anchor_feature_vector, all_shop_feature_vector, num_closest = max_k)
    
    # Select top max_k shop product
    all_shop_product   = list(np.array(all_shop_product)[list(cos_indices)])

    
    return all_shop_product


def find_accuracy_k_retrival(top_k_retrival):
    
    k_list = [1, 5, 10, 20, 50]
    k_result = {}
    
    for k in k_list:
        hit   = 0
        miss  = 0
        for item in top_k_retrival:
            product_id    = item['product_id']
            retrival_list = item['retrivals'][0:k]
            if product_id in retrival_list:
                hit += 1

            else:
                miss += 1


        k_result[str(k)] = {'hit':(hit/len(top_k_retrival)), 'miss':(miss/len(top_k_retrival))}
        
        
    return k_result

def find_feature_vector_bag(siamese_model, embedding_size, new_dataset_location, category, img_size):
    # Create feature vector for all required images
    fv_bag = {}
    img_key_holder = {'id':[], 'img':[]}
    
    list_of_all_images = glob(f"{new_dataset_location}/{category}/*/*/*")
    
    
    ind = 0
    for img_path in tqdm(list_of_all_images):

        img    = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), img_size)
    #     img_id = img_path.split('/')[-1]

        img_key_holder['id'].append(img_path)
        img_key_holder['img'].append(img)

        if (ind+1)%256==0:
            assert len(img_key_holder['id'])==256
            feature_vector  =  find_feature_vector(siamese_model, img_key_holder['img'], embedding_size)

            for k, fv in zip(img_key_holder['id'], feature_vector):
                fv_bag[str(k)] = fv


            img_key_holder = {'id':[], 'img':[]}


        ind += 1

    if len(img_key_holder['img']) > 0:
        feature_vector  =  find_feature_vector(siamese_model, img_key_holder['img'], embedding_size)     
        for k, fv in zip(img_key_holder['id'], feature_vector):
            fv_bag[str(k)] = fv

    return fv_bag