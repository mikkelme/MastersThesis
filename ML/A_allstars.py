from hypertuning import *


class A_allstar(Architectures):
    def common_settings(self):
        # Data outputs
        self.alpha = [[1/2, 1/10, 1/10], [1/10], [1/10, 1/10]]
        self.criterion_out_features = [['R', 'R', 'R'], ['R'], ['R', 'C']]
        self.keys = ['Ff_mean', 'Ff_max', 'contact', 'porosity', 'rupture_stretch', 'is_ruptured']
        self.model_out_features = [item for sublist in self.criterion_out_features for item in sublist]   

    
    # def A1(self): #C16C32C64D64
    #     """ Hanakate accelerated search """
    #     # Model
    #     model = VGGNet( name = 'C16C32C64D64',
    #                     mode = self.mode, 
    #                     input_num = 2, 
    #                     conv_layers = [(1, 16), (1, 32), (1, 64)], 
    #                     FC_layers = [(1, 64)],
    #                     out_features = self.model_out_features,
    #                     keys = self.keys,
    #                     batchnorm = self.batchnorm)
        
    #     # Criterion
    #     criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
    #     return model, criterion
    
    # def A2(self): #C16C32C64D512D128
    #     """ Hanakate accelerated search """
    #     # Model
    #     model = VGGNet( name = 'C16C32C64D512D128',
    #                     mode = self.mode, 
    #                     input_num = 2, 
    #                     conv_layers = [(1, 16), (1, 32), (1, 64)], 
    #                     FC_layers = [(1, 512), (1, 128)],
    #                     out_features = self.model_out_features,
    #                     keys = self.keys,
    #                     batchnorm = self.batchnorm)
        
    #     # Criterion
    #     criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
    #     return model, criterion
    
    # def A3(self): #C16C32C64C64D512D128
    #     # Model
    #     model = VGGNet( name = 'C16C32C64C64D512D128',
    #                     mode = self.mode, 
    #                     input_num = 2, 
    #                     conv_layers = [(1, 16), (1, 32), (1, 64), (1, 64)], 
    #                     FC_layers = [(1, 512), (1, 128)],
    #                     out_features = self.model_out_features,
    #                     keys = self.keys,
    #                     batchnorm = self.batchnorm)
        
    #     # Criterion
    #     criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
    #     return model, criterion
   
    # def A4(self): #C16C32C64C128D64D32D16
    #     # Model
    #     model = VGGNet( name = 'C16C32C64C128D64D32D16',
    #                     mode = self.mode, 
    #                     input_num = 2, 
    #                     conv_layers = [(1, 16), (1, 32), (1, 64), (1, 128)], 
    #                     FC_layers = [(1, 64), (1, 32), (1, 16)],
    #                     out_features = self.model_out_features,
    #                     keys = self.keys,
    #                     batchnorm = self.batchnorm)
        
    #     # Criterion
    #     criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
    #     return model, criterion
   
    def A5(self): #C32C64C128D128D64D32
        # Model
        model = VGGNet( name = 'C32C64C128D128D64D32',
                        mode = self.mode, 
                        input_num = 2, 
                        conv_layers = [(1, 32), (1, 64), (1, 128)], 
                        FC_layers = [(1, 128), (1, 64), (1, 32)],
                        out_features = self.model_out_features,
                        keys = self.keys,
                        batchnorm = self.batchnorm)
        
        # Criterion
        criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
        return model, criterion
    
    
    
if __name__ == '__main__':
    # root = '../Data/ML_data/' # Relative
    root = '/home/users/mikkelme/ML_data/' # Absolute cluster
    data_root = [root+'baseline', root+'popup', root+'honeycomb']
    
    ML_setting = {
        'use_gpu': True,
        'lr': 0.005,  # Learning rate
        'batchsize_train': 32,
        'batchsize_val': 64,
        'max_epochs': 1000,
        'max_file_num': None,
        'scheduler_stepsize': None, # 10
        'scheduler_factor': None # 0.3
    }
    
    
    
    A = A_allstar(mode = 0, batchnorm = True)
    train_architectures(A, data_root, ML_setting, save_folder = 'Hanakata')
    
    
    