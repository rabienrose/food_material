import sys
import config.chamo
import config.chamo_full_run
import data_preprocessing.default_preprocess
import data_preprocessing.test_preprocess
import net.vgg16
import net.mobilenet_v2
import loss.default_loss
import loss.entropy_loss
import accuracy.default_accuracy
import accuracy.multi_accuracy
import optimizer.default_opt
import eval.default_eval
import utils.data_helper

#config_name=sys.argv[1]
config_name='chamo'
print('choose config: '+config_name)
config_obj=None
if config_name=='chamo':
    config_obj=config.chamo.get_config()
elif config_name=='chamo_full_run':
    config_obj = config.chamo_full_run.get_config()

preprocess_name=config_obj.preprocess_type
preprocess_obj=None
test_preprocess_obj=None
if preprocess_name=='default':
    preprocess_obj=data_preprocessing.default_preprocess.default_preprocess(
        config_obj.tfrecord_addr,
        config_obj.batchsize,
        config_obj.class_num
    )

test_preprocess_obj=data_preprocessing.test_preprocess.test_preprocess(config_obj.tfrecord_test_addr, config_obj.class_num)
net_name=config_obj.net_type
net_obj=None
test_net_obj=None
if net_name=='vgg16':
    net_obj=net.vgg16.vgg16(True, 'vgg16', config_obj.class_num)
    test_net_obj=net.vgg16.vgg16(False, 'vgg16', config_obj.class_num)
elif net_name=='mobilenet_v2':
    net_obj = net.mobilenet_v2.mobilenet_v2(True, 'mobilenet_v2', config_obj.class_num)
    test_net_obj = net.mobilenet_v2.mobilenet_v2(True, 'mobilenet_v2', config_obj.class_num)


loss_name=config_obj.loss_type
loss_obj=None
if loss_name=='default':
    loss_obj=loss.default_loss.default_loss()
elif loss_name=='entropy_loss':
    loss_obj =loss.entropy_loss.entropy_loss()

accu_name=config_obj.accuracy_type
accu_obj=None
if accu_name=='default':
    accu_obj=accuracy.default_accuracy.default_accuracy()
elif accu_name=='multi':
    accu_obj=accuracy.multi_accuracy.multi_accuracy()

opt_name=config_obj.opt_type
opt_obj=None
if opt_name=='default':
    opt_obj=optimizer.default_opt.default_opt(
        config_obj.max_step,
        config_obj.debug_step_len,
        config_obj.result_addr,
        config_obj.stop_accu
    )

images, labels = preprocess_obj.def_preposess()
#utils.data_helper.check_imgs(images, labels)
images_test, labels_test = test_preprocess_obj.def_preposess()
net = net_obj.def_net(images)
net_test = test_net_obj.def_net(images_test)
loss = loss_obj.def_loss(net, labels)
test_accu = accu_obj.def_accuracy(net_test, labels_test)
opt_obj.run(loss, test_accu, config_obj.loading_his)
# if config_obj.is_training:
opt_obj.run(loss, test_accu, config_obj.loading_his)
# else:
#     eval_obj = eval.default_eval.default_eval(1000, config_obj.result_addr)
#     eval_obj.run(loss, test_accu)

