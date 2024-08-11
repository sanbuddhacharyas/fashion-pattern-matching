import json
import random

import tensorflow as tf
import tensorflow.compat.v1 as tf1

class Dataloader():
    def __init__(self, batch_size, val_batch_size, num_iter_per_style, img_size, filename_train, filename_val):
        self.filename_train      = filename_train
        self.filename_val        = filename_val
        self.batch_size          = batch_size
        self.img_size            = img_size
        self.val_batch_size      = val_batch_size
        self.num_iter_per_style  = num_iter_per_style
        self.data_loader()
        
    def dataset_generator(self, filename, batch_size, num_item_per_style):
    
        # load json file
        with open(filename, 'r') as f:
            training_files = json.load(f)

        # Loop until batch is possible
        while True:
            # break main loop if there is no any images left
            if len(training_files)==0:
                break

            # select available list of category
            cat_keys         = list(training_files.keys())
            selected_cat_key = random.choice(cat_keys)

            # select available list of style with in category
            all_styles   = training_files[selected_cat_key]
            style_list   = list(all_styles.keys())

            # shuffle all available styles
            random.shuffle(style_list)    

            selected_batches = []                    # store all the batches
            break_signal     = False                 # boolean required to stop loop once batch is completed

            for style_id in style_list:              # Loop through available styles
                # select items from style
                selected_style = all_styles[style_id]

                # shuffle the item from in selected style which is 
                random.shuffle(selected_style)
                
                # limit the maximum item from each style
                iterate         = (num_item_per_style if len(selected_style) > num_item_per_style else len(selected_style))
                
                # loop through each item within style
                for item in selected_style[:iterate]:
                    selected_batches.append(item)            # add item to batch
                    selected_style.remove(item)              # remove item from dictionary
                    if len(selected_batches)>= batch_size:   # batch is completed stop iteration through items
                        break_signal = True
                        break

                if len(selected_style)==0:                   # if empty value for style remove this key from dictionary
                    all_styles.pop(style_id, None)

                if break_signal:                             # if batch is completed stop iteration through style
                    break

            if len(all_styles) == 0:                         # if cateogory is empty remove cateogory key from dictionary
                training_files.pop(selected_cat_key, None)

            if len(selected_batches)==batch_size:            # only select the batches which are complete
                for item in selected_batches:
                    yield item

    def paras_data(self, img_path):

        img_path        = 'data/' + img_path
        class_label     = tf1.string_split([img_path],sep='/').values[-2]
        image           = tf1.image.decode_png(tf.io.read_file(img_path))
        image           = tf1.image.convert_image_dtype(image, tf.uint8)
        image           = tf1.image.resize(image, size = self.img_size )
       
        return image, class_label

    def train_preprocessing(self, image, class_label):
    
        #0.5 probability
        do_flip   = tf1.random_uniform([],0,1)

        #Horizantal flipping
        image  = tf1.cond(do_flip  > 0.5, lambda : tf1.image.flip_left_right(image),  lambda  : image)
        image.set_shape([self.img_size[0], self.img_size[1], 3])

        return (image, class_label)


    def test_preprocessing(self, image, class_label):
            
        image.set_shape([self.img_size[0], self.img_size[1], 3])

        return (image, class_label)
    
    def create_dataset(self, is_training=True):

        if is_training:
            loader   = tf.data.Dataset.from_generator(self.dataset_generator, args=[self.filename_train, self.batch_size, self.num_iter_per_style], output_types=tf.string, output_shapes = (),)
            
        else:
            loader   = tf.data.Dataset.from_generator(self.dataset_generator, args=[self.filename_val, self.val_batch_size, self.num_iter_per_style], output_types=tf.string, output_shapes = (),)
            
        loader   = loader.repeat()
        loader   = loader.map(self.paras_data)
        if is_training:
            loader   = loader.map(self.train_preprocessing)

        else:
            loader   = loader.map(self.test_preprocessing)
            
            
        loader   = loader.batch(self.batch_size)
        loader   = loader.prefetch(self.batch_size * 2)

        return loader


    def data_loader(self):

        self.train_dataset = self.create_dataset()
        self.test_dataset  = self.create_dataset(is_training = False)

        #create an iterator 
        self.train_iterator = tf1.data.make_one_shot_iterator(self.train_dataset)
        self.test_iterator  = tf1.data.make_one_shot_iterator(self.test_dataset)