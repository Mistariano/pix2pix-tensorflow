from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cv2 import resize, applyColorMap, COLORMAP_JET, cvtColor, COLOR_BGR2RGB
from PIL import Image
from pix2pix import *


def grad_cam_layer(discrim_out, conv_layer):
    """
    为模型添加grad_cam梯度求解层
    adapted from https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py

    :param discrim_out: 判别器输出。是一个1*30*30*1的张量
    :param conv_layer: 需要观察的卷积层
    :return: 需要被fetch的3个层：[conv_layer, norm_grads, loss]
    """

    print("Setting gradients to 1 for target class and rest to 0")
    loss = tf.reduce_mean(discrim_out)  #
    grads = tf.gradients(loss, conv_layer)[0]
    # Normalizing the gradients
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    return conv_layer, norm_grads, loss


def fetch_heatmap(fetch_layers, sess):
    """
    获得grad_cam求出的heatmap
    adapted from https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py

    :param fetch_layers: grad_cam_layer输出的三个tensor
    :param sess: 当前session
    :return: heatmap, [256,256,3], PIL格式(RGB)
    """

    output, grads_val, discriminate_out = sess.run(fetch_layers)
    print(discriminate_out)  # 判别器输出
    output = output[0]  # 4d -> 3d
    grads_val = grads_val[0]  # 4d -> 3d
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)
    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
        # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = resize(cam, (256, 256))

    cam3 = applyColorMap((255 * cam).astype('uint8'), COLORMAP_JET)  # 灰度 -> 热图
    cam3 = cvtColor(cam3, COLOR_BGR2RGB)  # cv2格式转PIL格式

    # Converting grayscale to 3-D,
    # old version
    #
    # //////////////////////////////////////////////
    # cam3 = np.expand_dims(cam, axis=2)
    # r = np.zeros(cam3.shape)
    # g = np.zeros(cam3.shape)
    # b = np.zeros(cam3.shape)
    # for i in range(cam3.shape[0]):
    #     for j in range(cam3.shape[1]):
    #         cur = cam3[i, j, 0]
    #         if cur < 0.5:
    #             b[i, j, 0] = 1. - cur / 0.5
    #             g[i, j, 0] = cur / 0.5
    #         elif 0.5 <= cur:
    #             g[i, j, 0] = 2. - cur / 0.5
    #             r[i, j, 0] = cur / 0.5 - 1
    # cam3 = np.concatenate([r, g, b], -1)

    return cam3


def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "gradcam":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    gc = grad_cam_layer(model.predict_fake, model.predict_fake_layers[-2])
    gc_2 = grad_cam_layer(model.predict_fake, model.predict_fake_layers[-3])
    gc_3 = grad_cam_layer(model.predict_fake, model.predict_fake_layers[-4])
    gc_real = grad_cam_layer(model.predict_real, model.predict_real_layers[-2])
    gc_real_2 = grad_cam_layer(model.predict_real, model.predict_real_layers[-3])
    gc_real_3 = grad_cam_layer(model.predict_real, model.predict_real_layers[-4])

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode in ["test", 'gradcam']:
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
                print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        if a.mode == 'gradcam':

            # 输出热图

            with tf.Graph().as_default():

                image_dir = os.path.join(a.output_dir, "images/")
                if not os.path.exists(image_dir):
                    os.mkdir(image_dir)

                cam = fetch_heatmap(gc, sess)
                Image.fromarray(cam).save(image_dir + 'fake_heat_-1.jpg')

                cam_real = fetch_heatmap(gc_real, sess)
                Image.fromarray(cam_real).save(image_dir + 'real_heat_-1.jpg')

                cam = fetch_heatmap(gc_2, sess)
                Image.fromarray(cam).save(image_dir + 'fake_heat_-2.jpg')

                cam_real = fetch_heatmap(gc_real_2, sess)
                Image.fromarray(cam_real).save(image_dir + 'real_heat_-2.jpg')

                cam = fetch_heatmap(gc_3, sess)
                Image.fromarray(cam).save(image_dir + 'fake_heat_-3.jpg')

                cam_real = fetch_heatmap(gc_real_3, sess)
                Image.fromarray(cam_real).save(image_dir + 'real_heat_-3.jpg')


if __name__ == '__main__':
    main()
