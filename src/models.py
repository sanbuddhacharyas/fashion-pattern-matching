import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras        import metrics

from tensorflow.keras.layers import Input, concatenate, Conv2D, Dense, BatchNormalization, Flatten, Layer, GlobalAveragePooling2D,AveragePooling2D, Activation
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

class DistanceLayer(Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self.batch_all_triplet_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.batch_all_triplet_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        
        return loss
    
    def _pairwise_distances(self, embeddings, squared=True):
        """Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = tf.linalg.diag_part(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.cast(tf.equal(distances, 0.0), dtype = tf.float32)
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances


    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal     = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j     = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k     = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k     = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j   = tf.expand_dims(label_equal, 2)
        i_equal_k   = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask
    
    
    def batch_all_triplet_loss(self, data):
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        input_image, labels = data  
        embeddings = self.siamese_network(input_image)
    
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask         = self._get_triplet_mask(labels)
        mask         = tf.cast(mask, dtype=tf.float32)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
    

# Atttention layer
def channel_attention(input_shape, reduction: int = 16, name: str = "") -> KM.Model:
    """channel attention model
    Args:
        features (int): number of features for incoming tensor
        reduction (int, optional): Reduction ratio for the MLP to squeeze information across channels. Defaults to 16.
        name (str, optional): Defaults to "".
    Returns:
        KM.Model: channelwise attention appllier model
    """
    features     = input_shape[-1]
    input_tensor = Input(shape=input_shape)

    # Average pool over a feature map across channels
    avg = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    
    # Max pool over a feature map across channels
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)

    # Number of features for middle layer of shared MLP
    reduced_features = int(features // reduction)

    dense1      = Dense(reduced_features)
    avg_reduced = dense1(avg)
    max_reduced = dense1(max_pool)

    dense2        = Dense(features)
    avg_attention = dense2(Activation("relu")(avg_reduced))
    max_attention = dense2(Activation("relu")(max_reduced))

    # Channel-wise attention
    overall_attention = Activation("sigmoid")(avg_attention + max_attention)

    return Model(
        inputs=input_tensor, outputs=input_tensor * overall_attention, name=name
    )


def spatial_attention(
    input_shape, kernel: int = 7 , bias: bool = False, name: str = "") -> Model:
    """spatial attention model
    Args:
        features (int): number of features for incoming tensor
        kernel (int): convolutional kernel size
        bias (bool, optional): whether to use bias in convolutional layer
        name (str, optional): Defaults to "".
    Returns:
        Model: spatial attention appllier model
    """

    input_tensor = Input(shape=input_shape)
    # Average pool across channels for a given spatial location
    avg = tf.reduce_mean(input_tensor, axis=[-1], keepdims=True)

    # Max pool across channels for a given spatial location
    max_pool = tf.reduce_max(input_tensor, axis=[-1], keepdims=True)

    concat_pool = tf.concat([avg, max_pool], axis=-1)

    # Attention for spatial locations
    conv = Conv2D(
        1, (kernel, kernel), strides=(1, 1), padding="same", use_bias=bias
    )(concat_pool)
    attention = Activation("sigmoid")(BatchNormalization()(conv))

    return Model(inputs=input_tensor, outputs=input_tensor * attention, name=name)


def cbam_block(
    input_shape,
    input_tensor: tf.Tensor,
    kernel: int = 7,
    spatial: bool = False,
    name: str = "",
) -> tf.Tensor:
    """Convolutional Block Attention Module as proposed by Woo et. al in
    CBAM: Convolutional Block Attention Module
    Args:
        input_tensor (tf.Tensor): feature tensor
        features (int): number of features for incoming layer
        kernel (int): kernel size for spatial attention module
        spatial (bool, optional): whether to apply spatial attention. Defaults to False.
                                False: equivalent to squueze and excitation block with both max and avg. pool.
        name (str, optional): Defaults to "".
    Returns:
        tf.Tensor: Attention-scaled feature tensor
    """

    out_tensor = channel_attention(input_shape, name=name + "chn")(input_tensor)
    if spatial:
        out_tensor = spatial_attention(input_shape, kernel=kernel, name=name + "spt")(
            out_tensor
        )
    return out_tensor
def spp_layer(input_, levels=(4, 2, 1), name='SPP_layer'):
    shape = input_.shape
    
    pyramid = []
    for n in levels:

        stride_1 = np.floor(float(shape[1] // n)).astype(np.int32)
        stride_2 = np.floor(float(shape[2] // n)).astype(np.int32)
        
        ksize_1 = stride_1 + (shape[1] % n)
        ksize_2 = stride_2 + (shape[2] % n)
        
        pool = tf.keras.layers.MaxPooling2D(pool_size=(ksize_1, ksize_2), \
                                             strides=(stride_1, stride_2), padding='valid')(input_)
    
        pyramid.append(tf.keras.layers.Flatten()(pool))
     
    spp_pool = tf.keras.layers.concatenate(pyramid, axis=1)
    
    return spp_pool


def encoder(img_size):
    input_tensor        = Input(shape=(img_size[0], img_size[1], 3))
    input_tensor_prepro = preprocess_input(input_tensor)

    base_model          = ResNet50(input_tensor=input_tensor_prepro, include_top=False, weights = 'imagenet')
    
    for layer in base_model.layers:
        if 'conv1' in layer.name or 'conv2' in layer.name:
            print(layer.name)
            base_model.get_layer(layer.name).trainable = False
            
        if layer.name == 'conv1_relu':
            attention_branch = layer.output
    
    # Applying CBAM in feature layer
    x             = Conv2D(4096, 1, padding="same", activation='relu', name='feature_vector')(base_model.output)
    embedding     = GlobalAveragePooling2D()(x)
    
    # Attention branch for lower features
    
    # Branch -1 
    #  input => (None, 128, 128, 64) | output => (None, 32, 32, 32)  downsample by 4
    downsample_layer   = AveragePooling2D(pool_size=(3, 3), strides =(4, 4), padding='same', name='AvgPool_1')(attention_branch) 
    
    # Conv block 1 input=> (None, 32, 32, 64) | output => (None, 16, 16, 128)
    conv_branch_1      = Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), padding='same', name = 'conv_branch_1')(downsample_layer)
    CBAM_layer_1       = cbam_block((int(img_size[0]/16), int(img_size[0]/16), 128), conv_branch_1, kernel = 7, spatial  = True,  name = 'CBAM_layer_1')
    
    # Conv block 2 input=> (None, 16, 16, 128) | output => (None, 8, 8, 128)
    conv_branch_2       = Conv2D(filters=128,kernel_size=(3, 3), strides=(2,2), padding='same', name ='conv_branch_2')(CBAM_layer_1)
    CBAM_layer_2        = cbam_block((int(img_size[0]/32), int(img_size[0]/32), 128), conv_branch_2, kernel = 7, spatial  = True, name = 'CBAM_layer_2')
    
    conv_branch_3       = Conv2D(filters=64,kernel_size=(3, 3), strides=(2,2), padding='same', name ='conv_branch_3')(CBAM_layer_2)
    CBAM_layer_3        = cbam_block((int(img_size[0]/64), int(img_size[0]/64), 64), conv_branch_3, kernel = 7, spatial  = True, name = 'CBAM_layer_')
    
    # Create linear embedding
    attention_embedding_1 = Flatten()(CBAM_layer_3)
    
    
    # Branch-2  downsample by 8
    downsample_layer   = AveragePooling2D(pool_size=(3, 3), strides =(8, 8), padding='same', name='AvgPool_2')(attention_branch) 
    
    # Conv block 1 input=> (None, 16, 16, 64) | output => (None, 8, 8, 128)
    conv_1_branch_1      = Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), padding='same',  name = 'conv_1_branch_1')(downsample_layer)
    CBAM_1_layer_1       = cbam_block((int(img_size[0]/32), int(img_size[0]/32), 128), conv_1_branch_1, kernel = 7, spatial  = True, name = 'CBAM_1_layer_1')
    
    # Conv block 2 input=> (None, 8, 8, 128) | output => (None, 4, 4, 64)
    conv_2_branch_2       = Conv2D(filters=64,kernel_size=(3, 3), strides=(2,2), padding='same', name = 'conv_2_branch_1')(CBAM_1_layer_1)
    CBAM_2_layer_2        = cbam_block((int(img_size[0]/64), int(img_size[0]/64), 64), conv_2_branch_2, kernel = 7, spatial  = True, name = 'CBAM_1_layer_2')
    
    attention_embedding_2 = Flatten()(CBAM_2_layer_2)
    
    # Concatenate output from branch attention and main branch
    linear_embedding    = concatenate([embedding, attention_embedding_1, attention_embedding_2], axis=-1)
    linear_embedding    = Dense(4096)(linear_embedding)
    
    # Create model
    encoder_model = Model(inputs=input_tensor, outputs=linear_embedding)

    return encoder_model

def create_model(img_size, margin, initial_learning_rate, decay_step):

    # define feature extraction model
    feature_extraction_model = encoder(img_size)

    # input layers
    input_img    = Input((img_size[0], img_size[1], 3), name = 'Input_image')
    
    # feature extraction from anchor and other images
    feature_embeddings = feature_extraction_model(input_img)
   
    # create model
    siamese_model = Model(inputs=input_img, outputs=feature_embeddings)

    # create new model with triplet loss function
    siamese_model = SiameseModel(siamese_model, margin=margin)

    #exponentialDecay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_step,
        decay_rate=0.96,
        staircase=True)
    
    #adam optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

    #compile model
    siamese_model.compile(optimizer=opt)
    
    return siamese_model