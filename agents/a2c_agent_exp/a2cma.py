import threading
import tensorflow as tf

import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../')))
from CP_CHESS.agents.a2c_agent_exp.model import Model
from CP_CHESS.agents.a2c_agent_exp.worker import Worker, LR


class A2CMAConfig(object):
    a_lr = 0.1
    c_lr = 0.2
    decay_interval = 50
    decay_rate = 0.5
    update_interval = 10


class A2CMultiAgent(object):
    def __init__(self, gpu_idx: int = None, n_worker: int = 1, model_dir: str = None, model_ver: int = None):
        self.global_net = Model(None, None, gpu_idx=gpu_idx, scope=Model.GLOBAL_SCOPE, global_net=None)
        if model_dir is None:
            model_dir = './model'
        if model_ver is None:
            model_ver = 0
        self.model_dir = model_dir
        self.model_ver = model_ver
        self.workers = []
        for i in range(n_worker):
            self.workers.append(Worker(self.global_net.graph, self.global_net.sess, gpu_idx=gpu_idx, name='W_{}'.format(i), global_net=self.global_net))
        for worker in self.workers:
            worker.env.opponent = self.global_net
        self.global_net.sess.run(self.global_net.init_op)
        self.load_model(model_dir, model_ver)
        self.lr = LR()
        self.lr.a_lr = A2CMAConfig.a_lr
        self.lr.c_lr = A2CMAConfig.c_lr

    def load_model(self, model_dir: str, model_ver: int) -> None:
        try:
            self.global_net.load('{}/model{}/model.ckpt'.format(model_dir, model_ver))
            print('model{} at {} was loaded'.format(model_dir, model_ver))
        except:
            print('model not found! save initial model instead')
            model_ver = 0
            save_path = self.global_net.save('{}/model{}/model.ckpt'.format(model_dir, model_ver))
            print('initial model saved at {}'.format(save_path))

    def train(self, n_eps: int = 100):
        for episode in range(n_eps):
            if episode % A2CMAConfig.decay_interval == 0:
                self.lr.a_lr *= A2CMAConfig.decay_rate
                self.lr.c_lr *= A2CMAConfig.decay_rate

            coord = tf.train.Coordinator()
            worker_threads = []
            for worker in self.workers:
                thread = threading.Thread(target=lambda: worker.work(coord, self.lr))
                thread.start()
                worker_threads.append(thread)
            coord.join(worker_threads)
            for worker in self.workers:
                if worker.update_feed_dict is not None:
                    worker.model.push_global(worker.update_feed_dict)

            if episode % A2CMAConfig.update_interval == 0:
                self.model_ver += 1
                save_path = self.global_net.save('{}/model{}/model.ckpt'.format(self.model_dir, self.model_ver))
                print('new model saved at {}'.format(save_path))
                for worker in self.workers:
                    worker.env.load_model(self.model_dir, self.model_ver)