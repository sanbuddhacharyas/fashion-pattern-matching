import os
import numpy as np
import yaml

from src.load_data import Dataloader
from src.custom_callback import CustomCallback
from src.utils import create_logger
from src.models import create_model

if __name__ == '__main__':

    with open ("config/config.yaml", 'r') as file:
        params = yaml.safe_load(file)

    os.makedirs(os.path.dirname(params.save_weights), exist_ok = True)
        
    steps_per_epoch  = np.ceil(params.total_train/params.batch_size).astype('int')
    validation_steps = np.ceil(params.total_test/params.batch_size).astype('int')

    model_name = f'Attention_branch_two_avgpool_flatten_{params.embedding_layer}_{params.img_size[0]}_{params.margin}_{params.batch_size}_online_all_batch'
    logger     = create_logger(model_name)

    os.makedirs(model_name, exist_ok=True)

    # log the parameters
    logger.info(f"total_train=>{params.total_train}")
    logger.info(f"total_test=>{params.total_test}")
    logger.info(f"Learning_rate=>{params.learning_rate}")
    logger.info(f"Batch_size=>{params.batch_size}")
    logger.info(f"Val_batch_size=>{params.val_batch_size}")
    logger.info(f"Image_size=>{img_size}")
    logger.info(f"Margin=>{margin}")
    logger.info(f"decay_step=>{decay_step}")
    logger.info(f"train_file_path={train_file_path}")
    logger.info(f"val_file_path=>{val_file_path}")
    logger.info(f"save_weights=>{save_weights}")

    siamese_model  = create_model(img_size, margin=margin, initial_learning_rate = learning_rate, decay_step=decay_step)
        
    if load_weights: 
        
        with open(epoch_saved_path, 'r') as f:
            iteration = int(f.readlines()[0])
            logger.info(f"Loaded iteration =>{iteration} | Epoch => {iteration//steps_per_epoch}")
            print(f"Loaded iteration =>{iteration} | Epoch => {iteration//steps_per_epoch}")
        

        logger.info(f"Model_weights has been loaded location=>{save_weights}")
        print(f"Model_weights has been loaded location==>{save_weights}")
        siamese_model.build(input_shape=(None, img_size[0], img_size[1], 3))
        siamese_model.load_weights(save_weights)
        
        tf.keras.backend.set_value(siamese_model.optimizer.iterations, iteration)
        val = tf.keras.backend.get_value(siamese_model.optimizer.iterations)
        print(val)
        

    logger.info(f"steps_per_epoch=>{steps_per_epoch}")
    logger.info(f"validation_steps=>{validation_steps}")

    loader = Dataloader(batch_size, val_batch_size, num_iter_per_style, img_size, train_file_path, val_file_path, '')