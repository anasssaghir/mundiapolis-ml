import tensorflow.keras as K

def save_model(network, filename):
    K.models.save_model(model=network,filepath=filename)
    return None


def load_model(filename):
    return K.models.load_model(filepath=filename)
