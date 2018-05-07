import numpy as np
from glob import glob
from collections import OrderedDict

import tensorflow as tf

from facenet.src import facenet
from memory import Memory


# For FaceNet
IMAGE_SIZE = 160
MODEL_ID = '20170512-110547'

# For reading images
CHARACTER_ORDER = (
    'geralt',
    'vesemir',

    'malczewski',
    'witkacy',

    'avatar_female',
    'avatar_male',

    'gollum',
    'smeagol',

    'durotan',
    'hulk',

    'thade',
    'cesar',
)


def read_images(image_size=IMAGE_SIZE, character_order=CHARACTER_ORDER):
    char_img_pairs = []
    prefix = 'facenet/datasets/faces_out/'
    for character in character_order:
        # TODO: path = glob(prefix + character + '/*png')
        print('YOU ARE USING JPEGS!!!')
        path = glob(prefix + character + '/*jpg')
        images = facenet.load_data(path, False, False, image_size)
        char_img = (character, images)
        char_img_pairs.append(char_img)
    return OrderedDict(char_img_pairs)


def get_num_images_per_character(images):
    num_images = np.unique(map(len, images.values()))
    assert len(num_images) == 1
    return num_images[0]


def get_image_size(images):
    return images.values()[0].shape[1]


def load_facenet(my_graph, my_sess, model=MODEL_ID):
    with my_graph.as_default():
        with my_sess.as_default():
            facenet.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')

    return images_placeholder, phase_train_placeholder, embeddings


def calc_embeddings(images, my_sess, images_placeholder, phase_train_placeholder, embedding_tensor):
    embeddings = OrderedDict()
    for character in images.keys():
        embeddings[character] = my_sess.run(embedding_tensor, {
            images_placeholder: images[character],
            phase_train_placeholder: False,
        })

    return embeddings


def prepare_batch(images, image_idx, train_phase=True):
    num_images_per_character = get_num_images_per_character(images)
    indexes = [image_idx] if train_phase else range(image_idx, num_images_per_character)

    batch_X = []
    batch_y = []
    for i in indexes:
        for character_id, character in enumerate(images.keys()):
            batch_X.append(images[character][image_idx])
            batch_y.append(character_id)

    return np.r_[batch_X], np.array(batch_y, dtype=np.int32)


def nearest_neighbor_predictions(query_embedds, all_embeddings, boundary_idx):
    """Find labels corresponding to obervations in `query_embeddings`.
    The idea is that we can use observations in `all_embeddings` but only up to
    index `boundary_idx` (inclusive)."""

    num_characters = len(all_embeddings)

    # Without any knowledge we migh as well throw a dice to make a prediction
    if boundary_idx == 0:
        return np.random.randint(0, num_characters, size=len(query_embedds))

    # We're looking for embedds that ...
    predictions = []
    for query in query_embedds:
        closest_cossims = [
            all_embeddings[char][:boundary_idx].dot(query).max()
            for char in all_embeddings.keys()
        ]
        predictions.append(np.argmax(closest_cossims))

    return np.array(predictions)


def eval_facenet_without_memory(embeddings):
    num_images_per_character = get_num_images_per_character(embeddings)

    results = {}
    for image_idx in range(num_images_per_character-1):
        # Prediction phase
        partial_results = {}
        batch_X, batch_y = prepare_batch(embeddings, image_idx, train_phase=False)
        partial_results['true'] = batch_y
        predictions = nearest_neighbor_predictions(batch_X, embeddings, image_idx)
        partial_results['pred'] = predictions
        results[image_idx] = partial_results

        # Note: there's no training phase
        # ...

    return results

def setup_memory(my_graph, embedding_tensor, learning_rate=1e-4):
    with my_graph.as_default():
        embedding_size = embedding_tensor.get_shape()[1]

        labels_placeholder = tf.placeholder(tf.int32, shape=[None])

        memory = Memory(
            key_dim=embedding_size,
            memory_size=2**5,
            vocab_size=120,
        )

        mem_var_init_op = tf.variables_initializer(var_list=[
            memory.mem_keys,
            memory.mem_vals,
            memory.mem_age,
            memory.recent_idx,
            memory.query_proj,
        ])

        closest_label_train, _, teacher_loss_train = memory.query(
            embedding_tensor, labels_placeholder, use_recent_idx=False)

        train_op = (tf.train
                    .GradientDescentOptimizer(learning_rate)
                    .minimize(teacher_loss_train))

        closest_label_pred, _, _ = memory.query(
            embedding_tensor, None, use_recent_idx=False)

    return mem_var_init_op, labels_placeholder, closest_label_pred, train_op


def train_and_eval_facenet_with_memory(images,
                                       my_sess,
                                       images_placeholder, phase_train_placeholder, labels_placeholder,
                                       mem_var_init_op, train_op, closest_label_pred):
    num_images_per_character = get_num_images_per_character(images)
    with my_sess.as_default():
        # Initialize the memory variables
        my_sess.run(mem_var_init_op)

        results = {}
        for image_idx in range(num_images_per_character-1):
            # Prediction phase
            partial_results = {}
            batch_X, batch_y = prepare_batch(images, image_idx, train_phase=False)
            partial_results['true'] = batch_y
            predictions = my_sess.run(closest_label_pred, {
                images_placeholder: batch_X,
                phase_train_placeholder: False,
            })
            partial_results['pred'] = predictions
            results[image_idx] = partial_results

            # Training phase
            batch_X, batch_y = prepare_batch(images, image_idx)
            feed_dict_train = {
                images_placeholder: batch_X,
                labels_placeholder: batch_y,
                phase_train_placeholder: False,  # TODO
            }

            my_sess.run(train_op, feed_dict_train)
            # return results

    return results


if __name__ == '__main__':
    images = read_images()
