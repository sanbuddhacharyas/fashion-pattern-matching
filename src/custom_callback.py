class CustomCallback(Callback):
    def __init__(self, logger, save_dir, epoch_saved_path):
        self.logger = logger
        self.save_dir = save_dir
   
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.logger.info(f"Itr=>{self.model.optimizer.iterations.numpy()} | Epoch=>{epoch}  | Train_loss=>{logs['loss']} | Validatition_loss=>{logs['val_loss']}")
        with open(self.save_dir, 'a') as f:
            f.write(f"Itr=>{self.model.optimizer.iterations.numpy()} | Epoch=>{epoch}  | Train_loss=>{logs['loss']} | Validatition_loss=>{logs['val_loss']}\n")

        with open(epoch_saved_path, 'w') as f:
            iteration = int(siamese_model.optimizer.iterations)
            f.write(str(iteration))
            
        print("Uploading Models in S3 Bucket")
        upload_file(save_weights, bucket, aws_save_weights)            # Upload saved weights
        upload_file(f'{model_name}/app.log', bucket, aws_logs_path)    # Upload log file
        upload_file(val_loss_save_dir, bucket, aws_val_loss_save_dir)  # Upload val text file
        upload_file(epoch_saved_path, bucket, aws_epoch_saved_path)    # Upload epoch .txt
            
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if (self.model.optimizer.iterations).numpy() % 100 == 0:
            self.logger.info(f"Itr=>{(self.model.optimizer.iterations).numpy()}  | Train_loss=>{logs['loss']}")