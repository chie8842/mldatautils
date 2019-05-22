import os
from keras.callbacks import Callback
from mldatautils.utils import get_jstdate_string
from mldatautils.utils import make_dirs
from mldatautils.utils import ModelCheckpoint
from hyperdash import Experiment

class Hyperdash(Callback):
    def __init__(self, entries, exp):
        super(Hyperdash, self).__init__()
        self.entries = entries
        self.exp = exp

    def on_epoch_end(self, epoch, logs=None):
        for entrie in self.entries:
            log = logs.get(entrie)
            if log is not None:
                self.exp.metric(entrie, log)

def prepare_experiment(exp_name, model_dir='models'):
    exp = Experiment(exp_name)
    model_dir = os.path.join(model_dir, exp_name, get_jstdate_string())
    checkpoint_filename = os.path.join(model_dir, 'checkpoint_{epoch:02d}-{val_loss:.2f}.hdf5')
    make_dirs(model_dir)
    hd_callback = Hyperdash(['val_acc', 'val_loss'], exp)
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_filename, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=1)
    return exp, hd_callback, checkpoint
