import sys
import config.chamo
import data_preprocessing.default_preprocess
import net.vgg16
import loss.default_loss
import optimizer.default_opt

config_name=sys.argv[1]
print(config_name)
config_obj=None
if config_name=='chamo':
    config_obj=config.chamo.get_config()

preprocess_name=config_obj.preprocess_type
preprocess_obj=None
if preprocess_name=='default':
    preprocess_obj=data_preprocessing.default_preprocess.default_preprocess(config_obj.tfrecord_addr)

net_name=config_obj.net_type
net_obj=None
if net_name=='vgg16':
    net_obj=net.vgg16.vgg16()

loss_name=config_obj.loss_type
loss_obj=None
if loss_name=='default':
    loss_obj=loss.default_loss.default_loss()

opt_name=config_obj.opt_type
opt_obj=None
if opt_name=='default':
    opt_obj=optimizer.default_opt.default_opt()

images, labels = preprocess_obj.def_preposess()
net = net_obj.def_net(images)
loss=loss_obj.def_loss(net, labels)
opt_obj.run(loss)

