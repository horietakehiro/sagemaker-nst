# -*- coding: utf-8 -*-
import argparse
import os
import socket
import json
from glob import glob
import tensorflow as tf
from PIL import Image
import itertools
import numpy as np


TB_DESCRIPTION_MD = """
|TRANSFER|C_W|S_W|T_W|LR|SRM|
|---|---|---|---|---|---|
|{transfer}|{c_w}|{s_w}|{t_w}|{lr}|{srm}|
"""

class NstEngine(tf.Module):
    def __init__(self, content_shape, args, name=None):
        """
        content_shape : コンテンツ画像のshape e.g. (1, 512, 512, 3)
        args : Namespaceオブジェクト。
        """
        super(NstEngine, self).__init__(name=name)

        # サンプノートブックの通り、VGG19を特徴量抽出器として利用する。
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet') 
        outputs = [
            vgg.get_layer(target_layer).output 
            for target_layer in self.content_layers + self.style_layers
        ]
        self.model = tf.keras.Model([vgg.input], outputs)
        self.model.trainable = False

        # tf.function内部では変数を宣言できないので、モデルの初期化時に
        # 変数も初期化しておく必要がある。
        self.content_image = tf.Variable(tf.zeros(content_shape), dtype=tf.float32)
        self.loss = tf.Variable(tf.zeros((1)), dtype=tf.float32)

        self.style_image = None
        self.content_image_org = None
        self.style_image_org = None
        self.content_target = None
        self.style_target = None


        self.epoch = int(args.EPOCH)
        # 後々チューニング対象になるハイパーパラメータ達
        learning_rate = float(os.environ.get("SM_HP_LEARNING_RATE", args.LEARNING_RATE))
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.99,
            epsilon=0.1,
        )
        self.content_weights = float(os.environ.get("SM_HP_CONTENT_WEIGHTS", args.CONTENT_WEIGHTS))
        self.style_weights = float(os.environ.get("SM_HP_STYLE_WEIGHTS", args.STYLE_WEIGHTS))
        self.total_variation_weights = float(os.environ.get("SM_HP_TOTAL_VARIATION_WEIGHTS", args.TOTAL_VARIATION_WEIGHTS))


    @tf.function
    def fit(self, content, style, content_org):
        """
        args:
            - content : 更新対象のコンテンツ画像 shape : (1, height, width, 3)
            - style : スタイル画像 shape : (1, height, width, 3)
            - content_org : オリジナルのコンテンツ画像 shape : (1, height, width, 3)

        return : 
            - スタイル画像の画風で更新されたコンテンツ画像 shape : (1, height, width, 3)
            - loss値 shape : (1)
        """

        self.content_image_org = content_org
        self.style_image_org = style

        self.content_image.assign(content)
        self.style_image = style

        self.content_target = self.call(self.content_image_org)['content']
        self.style_target = self.call(self.style_image_org)['style']

        for e in tf.range(self.epoch):
            self.loss.assign([self.step()])
        
        return self.content_image, self.loss
    
    @tf.function
    def step(self):
        """
        以降の関数はサンプルノートブックの処理を流用。
        詳細については省略
        """
        with tf.GradientTape() as tape:
            outputs = self.call(self.content_image)
            loss = self._calc_style_content_loss(outputs)
            loss += self.total_variation_weights*self._total_variation_loss()

        grad = tape.gradient(loss, self.content_image)
        self.optimizer.apply_gradients([(grad, self.content_image)])
        self.content_image.assign(self._clip_0_1())

        return loss

    def _calc_style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([
            tf.reduce_mean((style_outputs[name] - self.style_target[name])**2)
            for name in style_outputs.keys()
        ])
        style_loss *= self.style_weights / len(self.style_layers)

        content_loss = tf.add_n([
            tf.reduce_mean((content_outputs[name] - self.content_target[name])**2)
            for name in content_outputs.keys()
        ])
        content_loss *= self.content_weights / len(self.content_layers)

        loss = style_loss + content_loss
        return loss

    def _total_variation_loss(self):
        x_deltas, y_deltas = self._high_pass_x_y()
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    def _clip_0_1(self):
        clipped = tf.clip_by_value(
            self.content_image,clip_value_min=0.0, clip_value_max=1.0
        )
        return clipped

    def _high_pass_x_y(self):
        x_var = self.content_image[:, :, 1:, :] - self.content_image[:, :, :-1, :]
        y_var = self.content_image[:, 1:, :, :] - self.content_image[:, :-1, :, :]
        return x_var, y_var


    def call(self, input_image):
        input_image = input_image * 255.
        image = tf.keras.applications.vgg19.preprocess_input(input_image)
        outputs = self.model(image)
        
        content_outputs = outputs[:len(self.content_layers)]
        style_outputs = outputs[len(self.content_layers):]

        style_matrix = self._calc_gram_matrix(style_outputs)

        style_dict = {
            name: output 
            for name, output in zip(self.style_layers, style_matrix)
        }
        content_dict = {
            name: output 
            for name, output in zip(self.content_layers, content_outputs)
        }

        return {'style' : style_dict, 'content' : content_dict}

    def _calc_gram_matrix(self, input_tensors):
        results = []
        for input_tensor in input_tensors:
            result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
            input_shape = tf.shape(input_tensor)
            num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
            results.append(result / num_locations)
        return results


def _parse_args():
    """
    return : Tuple(Namespace, List[str])
    """
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', "models"))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', "train"))
    parser.add_argument('--sm-output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', "outputs"))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS', "{}")))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', socket.gethostname()))


    # custome hyperparameters that may be tuned
    parser.add_argument('--CONTENT_WEIGHTS', type=float, default=10000)
    parser.add_argument('--STYLE_WEIGHTS', type=float, default=0.01)
    parser.add_argument('--TOTAL_VARIATION_WEIGHTS', type=float, default=30)
    parser.add_argument("--LEARNING_RATE", type=float, default=0.02)
    parser.add_argument("--STYLE_RESIZE_METHOD", type=str, default="original")

    # other custome hyperparameters that are not be tuned
    parser.add_argument('--EPOCH', type=int, default=20)
    parser.add_argument('--STEP', type=int, default=25)
    parser.add_argument("--MAX_IMAGE_SIZE", type=int, default=512)
    parser.add_argument("--TB_BUCKET", type=str, default="")
    parser.add_argument("--MAX_TRIAL", type=int, default=25)

    return parser.parse_known_args()

def _load_and_preprocess_content_image(image_path, args):
    """
    return the image of:
        shape : [1, max(MAX_IMAGE_SIZE, original's height), max(MAX_IMAGE_SIZE, original's width), 3]
        scale : 0~1.
        dtype : float32
    """
    image = Image.open(image_path)
    # force to convert to RGB(3 channel)
    image = image.convert("RGB")

    # limit the image size to args.IMAGE_MAX_SIZE
    current_size = image.size
    if args.MAX_IMAGE_SIZE < max(current_size):
        scale = args.MAX_IMAGE_SIZE / max(current_size)
        new_size = (round(s * scale) for s in current_size)
        image = image.resize(new_size)

    # convert dtype
    image = np.array(image, dtype=np.float32)
    # scale value
    image = image / image.max(axis=None)
    # add new asix
    image = image[np.newaxis, ]

    return image


def _load_and_preprocess_style_image(image_path, args, content_shape):
    """
    content_shape : (1, ???, ???, 3)

    return the image of:
        shape : one of them below
            - original : [1, max(MAX_IMAGE_SIZE, original's height), max(MAX_IMAGE_SIZE, original's width), 3]
            - imagenet : [1, 224, 224, 3]
            - content  : [1, same_as_content_shape, 3]
            - medium   : [1, 512, 512, 3]
        scale : 0~1.
        dtype : float32
    """
    image = Image.open(image_path)
    # force to convert to RGB(3 channel)
    image = image.convert("RGB")

    resize_method = os.environ.get("SM_HP_STYLE_RESIZE_METHOD", args.STYLE_RESIZE_METHOD)
    if resize_method == "original":
        # limit the image size to args.IMAGE_MAX_SIZE
        current_size = image.size
        if args.MAX_IMAGE_SIZE < max(current_size):
            scale = args.MAX_IMAGE_SIZE / max(current_size)
            new_size = (round(s * scale) for s in current_size)
            image = image.resize(new_size)
    elif resize_method == "imagenet":
        # resize to (224, 224) - original size of imagenet
        image = image.resize((224, 224))
    elif resize_method == "content":
        # resize to the same size as content_image
        # note : image.resize((width, height)), while conteent_shape is (1, height, width, 3)
        h, w = content_shape[1:3]
        image = image.resize((w, h))
    elif resize_method == "medium":
        # resize to (512, 512)
        image = image.resize((512, 512))


    # convert dtype
    image = np.array(image, dtype=np.float32)
    # scale value
    image = image / image.max(axis=None)
    # add new asix
    image = image[np.newaxis, ]

    return image


def postprocess_image(transfer, transfer_path):
    """
    save the transfer resutl
    """
    # reshape to (X, X, 3)
    transfer = transfer[0, :, :, :]

    # scale to 0 ~ 255
    transfer = transfer * 255.

    # convert dtype into np.uint8
    transfer = transfer.astype(np.uint8)

    # convert into PIL image
    transfer = Image.fromarray(transfer, mode="RGB")

    # save
    if not os.path.exists(os.path.dirname(transfer_path)):
        os.mkdir(os.path.dirname(transfer_path))
    transfer.save(transfer_path)

    return transfer_path

def _concat_all_images(content, style, transfer):
    """
    returned shape : [3, max(all of images' length), max(all of images' length), 3]
    """
    max_length = np.max((content.shape, style.shape))
    c_pad = (
        (0, 0), 
        (max_length - content.shape[1], 0), 
        (max_length - content.shape[2], 0), 
        (0, 0)
    )
    s_pad = (
        (0, 0), 
        (max_length - style.shape[1], 0), 
        (max_length - style.shape[2], 0), 
        (0, 0)
    )
    c_padded = np.pad(content, c_pad)
    s_padded = np.pad(style, s_pad)
    t_padded = np.pad(transfer, c_pad)

    return np.concatenate([c_padded, s_padded, t_padded])

def main(content_path, style_path, args):

    transfer_path = os.path.join(
        args.train, "images", "transfer",
        "{}_{}".format(
            os.path.basename(content_path).split(".")[0], os.path.basename(style_path)
        )
    )
    print("transfer start : {}".format(os.path.basename(transfer_path)))

    # load and preprocess images
    content = _load_and_preprocess_content_image(content_path, args)
    style = _load_and_preprocess_style_image(style_path, args, content.shape)
    content_org = _load_and_preprocess_content_image(content_path, args)

    print("content shape : ", content.shape)
    print("style shape   : ", style.shape)

    # setup tensorboard logging
    if args.TB_BUCKET.startswith("s3://"):
        # e.g.
        # SM_HP_MODEL_DIR=s3://sagemaker-ap-northeast-1-XXXXXXXXXXXX/tensorflow-training-2020-11-03-07-12-14-837/model
        # or
        # SM_HP_MODEL_DIR=s3://sagemaker-ap-northeast-1-XXXXXXXXXXXX/tensorflow-training-2020-11-03-07-29-02-024/model/tensorflow-training-201103-1629-003-608b2aac/model
        prefix = os.environ.get("SM_HP_MODEL_DIR").split("/")[-2]
        tb_writer = tf.summary.create_file_writer(
            os.path.join(args.TB_BUCKET, prefix, os.path.basename(transfer_path).split(".")[0])
        )
    elif args.TB_BUCKET != "":
        tb_writer = tf.summary.create_file_writer(args.TB_BUCKET)
    else:
        tb_writer = None

    # initiate the model
    model = NstEngine(content.shape, args)
    # execute fit(looping)
    description_md = TB_DESCRIPTION_MD.format(
        transfer=os.path.basename(transfer_path),
        c_w=args.CONTENT_WEIGHTS,
        s_w=args.STYLE_WEIGHTS,
        t_w=args.TOTAL_VARIATION_WEIGHTS,
        lr=args.LEARNING_RATE,
        srm=args.STYLE_RESIZE_METHOD,
    )

    for step in range(1, args.STEP+1):
        transfer, loss = model.fit(content, style, content_org)
        content = transfer.numpy()
        # logging for tensorboard
        if tb_writer is not None:
            with tb_writer.as_default():
                tf.summary.scalar("loss", loss.numpy()[0], step, description=description_md)
                images = _concat_all_images(content_org, style, transfer.numpy())
                tf.summary.image("transfer image", images, step, description=description_md)
        print("step : {}/{} has done.".format(step, args.STEP))

    # save transfer image
    _ = postprocess_image(transfer.numpy(), transfer_path)

    # save the model just one time
    export_dir=os.path.join(args.sm_model_dir, "1")
    if not os.path.exists(export_dir):
        model = NstEngine(content.shape, args)
        fit = model.fit.get_concrete_function(
            tf.TensorSpec([None, None, None, None], tf.float32, name="content"),
            tf.TensorSpec([None, None, None, None], tf.float32, name="style"),
            tf.TensorSpec([None, None, None, None], tf.float32, name="transfer"),
        )
        tf.saved_model.save(
            obj=model, 
            export_dir=export_dir,
            signatures=fit
        )

    print("transfer end with : {}".format(os.path.basename(transfer_path)))

    return loss.numpy()[0]

def _get_image_path_list(args):
    content_path_list = glob(os.path.join(
        args.train, "images", "content", "*"
    ))
    style_path_list = glob(os.path.join(
        args.train, "images", "style", "*"
    ))

    return list(itertools.product(content_path_list, style_path_list))


if __name__ =='__main__':

    # parse arguments
    args, _ = _parse_args()
    print(args)

    # get all images' path as list
    image_path_list = _get_image_path_list(args)
    print("All image pairs : {}".format(image_path_list))

    # transfer with all image pairs
    loss_list = np.zeros(args.MAX_TRIAL, dtype="f")
    for i, (content_path, style_path) in enumerate(image_path_list):
        if i == args.MAX_TRIAL:
            break
        loss = main(content_path, style_path, args)
        loss_list[i] = loss

    print("FinalMeanLoss={}".format(loss_list.mean()))

