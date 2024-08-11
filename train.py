import os
import numpy as np
import yaml
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

# source files
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
    logger.info(f"Image_size=>{params.img_size}")
    logger.info(f"Margin=>{params.margin}")
    logger.info(f"decay_step=>{params.decay_step}")
    logger.info(f"train_file_path={params.train_file_path}")
    logger.info(f"val_file_path=>{params.val_file_path}")
    logger.info(f"save_weights=>{params.save_weights}")

    # Create a siamese model
    siamese_model  = create_model(params.img_size, margin=params.margin, initial_learning_rate = params.learning_rate, decay_step=params.decay_step)
        
    if params.load_weights: 
        
        with open(params.epoch_saved_path, 'r') as f:
            iteration = int(f.readlines()[0])
            logger.info(f"Loaded iteration =>{iteration} | Epoch => {iteration//steps_per_epoch}")
        
        logger.info(f"Model_weights has been loaded location=>{params.save_weights}")
        
        siamese_model.build(input_shape=(None, params.img_size[0], params.img_size[1], 3))
        siamese_model.load_weights(params.save_weights)
        
        tf.keras.backend.set_value(siamese_model.optimizer.iterations, iteration)
        val = tf.keras.backend.get_value(siamese_model.optimizer.iterations)
        

    logger.info(f"steps_per_epoch=>{steps_per_epoch}")
    logger.info(f"validation_steps=>{validation_steps}")

    # Load Dataset
    loader = Dataloader(params.batch_size, params.val_batch_size, params.num_iter_per_style, params.img_size, params.train_file_path, params.val_file_path)

    # create checkpoint object
    checkpoint_callback = ModelCheckpoint(filepath = params.save_weights, monitor='val_loss', verbose=1, save_best_only=False, mode='min',save_weights_only=True)

    # Train the Model
    history = siamese_model.fit(loader.train_iterator,
                            validation_data=loader.test_iterator,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps = validation_steps,
                            epochs=9, workers=0, use_multiprocessing=False, callbacks=[checkpoint_callback, CustomCallback(logger, params.val_loss_save_dir, params.epoch_saved_path)])
