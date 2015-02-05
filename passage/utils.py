import numpy as np
import theano
import theano.tensor as T
import cPickle

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data]) 

def iter_indices(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        yield b

def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def case_insensitive_import(module, name):
    mapping = dict((k.lower(), k) for k in dir(module))
    return getattr(module, mapping[name.lower()])

def load(path):
    import layers
    import models
    model = cPickle.load(open(path))
    model_class = getattr(models, model['model'])
    model['config']['layers'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layers']]
    model = model_class(**model['config'])
    return model

def save(model, path):
    layer_configs = []
    for layer in model.layers:
        layer_config = layer.settings
        layer_name = layer.__class__.__name__
        weights = [p.get_value() for p in layer.params]
        layer_config['weights'] = weights
        layer_configs.append({'layer':layer_name, 'config':layer_config})
    model.settings['layers'] = layer_configs
    serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
    cPickle.dump(serializable_model, open(path, 'wb'))