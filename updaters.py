import chainer
import six


class VQVAE_StandardUpdater(chainer.training.StandardUpdater):
    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        loss1, loss2, loss3 = loss_func(*in_arrays)
        optimizer.target.cleargrads()
        loss1.backwards()
        optimizer.target.vq.cleargrads()
        loss2.backwards()
        loss3.backwards()
        optimizer.update()


class VQVAE_ParallelUpdater(chainer.training.ParallelUpdater):
    def update_core(self):
        optimizer = self.get_optimizer('main')
        model_main = optimizer.target
        models_others = {k: v for k, v in self._models.items()
                         if v is not model_main}

        batch = self.get_iterator('main').next()

        #
        # Split the batch to sub-batches.
        #
        n = len(self._models)
        in_arrays_list = {}
        for i, key in enumerate(six.iterkeys(self._models)):
            in_arrays_list[key] = self.converter(
                batch[i::n], self._devices[key])

        # For reducing memory
        for model in six.itervalues(self._models):
            model.cleargrads()

        loss1s = []
        loss2s = []
        loss3s = []
        for model_key, model in six.iteritems(self._models):
            in_arrays = in_arrays_list[model_key]
            loss_func = self.loss_func or model

            with chainer.function.force_backprop_mode():
                loss1, loss2, loss3 = loss_func(*in_arrays)
            loss1s.append(loss1)
            loss2s.append(loss2)
            loss3s.append(loss3)

        # For _uninitialized_params
        for model in six.itervalues(self._models):
            model.cleargrads()

        for loss1 in loss1s:
            loss1.backward()

        for model in six.itervalues(self._models):
            model.vq.cleargrads()

        for loss2, loss3 in zip(loss2s, loss3s):
            loss2.backward()
            loss3.backward()

        for model in six.itervalues(models_others):
            model_main.addgrads(model)

        optimizer.update()

        for model in six.itervalues(models_others):
            model.copyparams(model_main)
