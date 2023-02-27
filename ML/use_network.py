

def config_profile(model):
    pass


def load_model(path, use_gpu = False):
    
    if use_gpu:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    return model

if __name__ == '__main__':
    model = load_model = 'test100_model_dict_state')
    