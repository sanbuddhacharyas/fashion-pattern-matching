import os
import json
import yaml
from glob import glob
from tqdm import tqdm
from tensorflow.keras.models import load_model

from src.models import create_model
from src.utils import find_feature_vector_bag, find_top_k_retrival, find_accuracy_k_retrival

if __name__ == '__main__':

    # Set model params
    model_name           = 'PARSNet'
    test_dataset_file    = 'data/test_dataset/fashiondataset_triplet_test_with_hard.txt'
    dataset_root_dir     = 'data/test_dataset/'
    new_dataset_location = f'{dataset_root_dir}/fasion_dataset_similar_pair_croped_validation'
    model_saved_path     = '/nfs/stak/users/buddhacs/hpc-share/personal_project/fashion-pattern-matching/weights/Attention_branch_two_avgpool_flatten_GlobalAvg_320_1.5_4_online_all_batch_model.h5'
    list_of_all_images   = glob(f"{new_dataset_location}/*/*/*/*")
    embedding_size       = 4096
    
    # Load params:
    with open('config/config.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Load Model
    siamese_model        = create_model(params['img_size'], margin=params['margin'], initial_learning_rate = eval(params['learning_rate']), decay_step=params['decay_step'])
    siamese_model.build((None, params['img_size'][0], params['img_size'][1], 3))
    siamese_model.load_weights(model_saved_path)

    with open(test_dataset_file, 'r') as f:
        list_to_test = [i.replace('\n', '')for i in f.readlines()]

    # Iterate through each category
    list_of_categories    = [i.split('/')[-1] for i in sorted(glob(new_dataset_location+'/*'))]  # List of available categories
    total_result          = {}

    for cat_num, category in enumerate(list_of_categories):
        print(category)
        list_of_style_per_category = sorted(glob(f"{new_dataset_location}/{category}/*"))  # All the availble style per category
        list_of_all_shop_ads       =  list_of_all_images = glob(f"{new_dataset_location}/{category}/*/shop/*")
        top_k_retrival             = []
        fv_bag                     = find_feature_vector_bag(siamese_model, embedding_size, new_dataset_location, category, params['img_size'])
        
        # Iterate through each style and pich consumer/user clothes
        style_num = 0
        for style_path in tqdm(list_of_style_per_category):
            list_of_users_in_style     =  glob(f"{style_path}/user/*")
            list_of_shop_ads_in_style  =  glob(f"{style_path}/shop/*")

            num_shop_ads = len(list_of_shop_ads_in_style)

            if num_shop_ads==0:
                continue

            # Loop through each user
            if len(list_of_users_in_style)>=1:
                for user_num, user_path in enumerate(list_of_users_in_style):

                    img_id     = user_path.split('/')[-1].split('.')[0]
                    product_id = user_path.split('/')[-3].split('_')[0]

                    retrival_product = find_top_k_retrival(user_path, list_of_all_shop_ads, fv_bag)
                    top_k_retrival.append({'img_id':img_id, 'product_id':product_id, 'retrivals':retrival_product})
                    
        if len(top_k_retrival)==0:
            continue
            
    
        k_results                   = find_accuracy_k_retrival(top_k_retrival)
        total_result[str(category)] = k_results
        print(k_results)
        
        # Clear objects
        import gc
        gc.collect()

    os.makedirs(f'output/{model_name}')
    with open(f'./output/{model_name}/top_20_retrival.json', 'w') as f:
        json.dump(total_result, f)