from argparse import Action
import tensorflow as tf
import glob
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras import layers, datasets, optimizers, losses, Sequential, Model


# 需要放在tensorflow调用前 , 动态显存,不要全部占用
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Generator(Model):
    # 生成器网络类
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = layers.Conv2DTranspose(
            filter * 8, 4, 1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(
            filter * 4, 4, 2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(
            filter * 2, 4, 2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2DTranspose(
            filter * 1, 4, 2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2DTranspose(3, 4, 2, 'same', use_bias=False)

    def call(self, inputs, training=None):
        x = inputs
        x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        x = tf.nn.leaky_relu(x)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        x = self.conv5(x)
        x = tf.tanh(x)
        return x


class Discriminator(Model):
    # 判别器类
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(filter, 4, 2, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filter * 2, 4, 2, 'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filter * 4, 4, 2, 'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(filter * 8, 3, 1, 'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(filter * 16, 3, 1, 'valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        self.pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def create_data():
    # 加载所有图片的路径
    img_path_arr = glob.glob(r'.\faces\*.jpg')
    all_image_paths = [str(path) for path in list(img_path_arr)]

    def handler(img_path):
        image = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [64, 64])
        return tf.cast(tf.cast(image, dtype=tf.float64) / 127.5 - 1, dtype=tf.float64)

    return tf.data.Dataset.from_tensor_slices(all_image_paths).map(handler).batch(batch_size)


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img
    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate(
                (single_row, preprocesed[b, :, :, :]), axis=1)
        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            # reset single row
            single_row = np.array([])
    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


batch_size = 8 * 12
filter = 64
learning_rate = 0.002    # 学习率
epochs = 30000
index = 0
z_dim = 100
if __name__ == '__main__':
    tf.test.is_gpu_available()
    # 创建数据
    np_image_arr = create_data()
    # 生成器,根据一些随机数,生成一个图像,输入4,100 维,4表示个数,100是随机数
    generator = Generator()
    generator.build(input_shape=(4, z_dim))
    # 判断器,判断是真还是假
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))
    # 确定学习率
    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    is_training = True
    for epoch in range(epochs):
        # 生成随机数,以供生成器生成
        uniform_data = tf.random.uniform(
            [batch_size, z_dim], minval=-1., maxval=1.)
        for step, true_img in enumerate(np_image_arr):
            # 虽然每次都会拉取 batch_size 个参数,但是最后一批数据不足 batch_size 时,将会少于
            cut_uniform_data = uniform_data[0: true_img.shape[0]]
            # 训练判断器
            with tf.GradientTape() as tape:
                # 生成器生成图片
                generator_fake_img = generator(cut_uniform_data, is_training)
                # 判断器判断
                # 判断生成器生成的假图片,判断器判断对错的结果
                fake_img_true_or_false = discriminator(
                    generator_fake_img, is_training)
                # 判断生成器生成的真图片,判断器判断对错的结果
                true_img_true_or_false = discriminator(true_img, is_training)
                # 计算损失
                # 判断假图片判断错误的损失
                fake_img_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_img_true_or_false,
                                                                        labels=tf.zeros_like(fake_img_true_or_false))
                fake_img_loss = tf.reduce_mean(fake_img_loss)
                # 判断真图片判断错误的损失
                true_img__loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=true_img_true_or_false,
                                                                         labels=tf.ones_like(true_img_true_or_false))
                true_img__loss = tf.reduce_mean(true_img__loss)
                d_loss = fake_img_loss + true_img__loss
            # 更新调参
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(
                zip(grads, discriminator.trainable_variables))
            # 训练生成器
            with tf.GradientTape() as tape:
                # 生成器生成假图
                fake_image = generator(cut_uniform_data, is_training)
                # 判断器判断假图中被判断为真的图片
                fake_image_true_or_false = discriminator(
                    fake_image, is_training)
                # 依据,生成器生成的假图片,成功欺骗判断器判断为真
                fake_image_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_image_true_or_false,
                                                                          labels=tf.ones_like(fake_image_true_or_false))
                g_loss = tf.reduce_mean(fake_image_loss)
            # 更新调参
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(
                zip(grads, generator.trainable_variables))
        # 每次结束,打印一下损失
        print(epoch, '判断器损失:', float(d_loss),
              '生成器损失:', float(g_loss))

        # 准备生成演示图片
        uniform_data = tf.random.uniform([100, z_dim])
        # 生成假图片
        fake_image = generator(uniform_data, training=False)
        # 加载保存路径
        img_path = os.path.join('images', 'gan-%d.png' % index)
        index += 1
        # 处理图片,并保存
        save_result(fake_image.numpy(), 10, img_path, color_mode='P')
