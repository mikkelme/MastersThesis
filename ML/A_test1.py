from hypertuning import *


class A_test(Architectures):
    def common_settings(self):
        # Data outputs
        self.alpha = [[1/2, 1/10, 1/10], [1/10], [1/10, 1/10]]
        self.criterion_out_features = [['R', 'R', 'R'], ['R'], ['R', 'C']]
        self.keys = ['Ff_mean', 'Ff_max', 'contact', 'porosity', 'rupture_stretch', 'is_ruptured']
        self.model_out_features = [item for sublist in self.criterion_out_features for item in sublist]   
    
    # def A1(self): #C16D16"
    #     # Model
    #     model = VGGNet( name = 'C16D16',
    #                     mode = self.mode, 
    #                     input_num = 2, 
    #                     conv_layers = [(1, 16)], 
    #                     FC_layers = [(1, 16)],
    #                     out_features = self.model_out_features,
    #                     keys = self.keys,
    #                     batchnorm = self.batchnorm)
        
    #     # Criterion
    #     criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
    #     return model, criterion

    # def A2(self): #C32D32"
    #     # Model
    #     model = VGGNet( name = 'C32D32',
    #                     mode = self.mode, 
    #                     input_num = 2, 
    #                     conv_layers = [(1, 32)], 
    #                     FC_layers = [(1, 32)],
    #                     out_features = self.model_out_features,
    #                     keys = self.keys,
    #                     batchnorm = self.batchnorm)
        
    #     # Criterion
    #     criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
    #     return model, criterion

    def A3(self): #C8C16D16D8"
        # Model
        model = VGGNet( name = 'C8C16D16D8',
                        mode = self.mode, 
                        input_num = 2, 
                        conv_layers = [(1, 8), (1, 16)], 
                        FC_layers = [(1, 16), (1, 8)],
                        out_features = self.model_out_features,
                        keys = self.keys,
                        batchnorm = self.batchnorm)
        
        # Criterion
        criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
        return model, criterion
    
    def A4(self): #C16C16D16D16"
        # Model
        model = VGGNet( name = 'C16C16D16D16',
                        mode = self.mode, 
                        input_num = 2, 
                        conv_layers = [(1, 16), (1, 16)], 
                        FC_layers = [(1, 16), (1, 16)],
                        out_features = self.model_out_features,
                        keys = self.keys,
                        batchnorm = self.batchnorm)
        
        # Criterion
        criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
        return model, criterion
    
    def A5(self): #C16C32D32D16"
        # Model
        model = VGGNet( name = 'C16C32D32D16',
                        mode = self.mode, 
                        input_num = 2, 
                        conv_layers = [(1, 16), (1, 32)], 
                        FC_layers = [(1, 32), (1, 16)],
                        out_features = self.model_out_features,
                        keys = self.keys,
                        batchnorm = self.batchnorm)
        
        # Criterion
        criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
        return model, criterion
    
    def A6(self): #C16C32C32D32D32D16"
        # Model
        model = VGGNet( name = 'C16C32C32D32D32D16',
                        mode = self.mode, 
                        input_num = 2, 
                        conv_layers = [(1, 16), (1, 32), (1, 32)], 
                        FC_layers = [(1, 32), (1, 32), (1, 16)],
                        out_features = self.model_out_features,
                        keys = self.keys,
                        batchnorm = self.batchnorm)
        
        # Criterion
        criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
        return model, criterion
    
    def A7(self): #C8C16C32C64D32D16D8"
        # Model
        model = VGGNet( name = 'C8C16C32C64D32D16D8',
                        mode = self.mode, 
                        input_num = 2, 
                        conv_layers = [(1, 8), (1, 16), (1, 32), (1, 64)], 
                        FC_layers = [(1, 32), (1, 16), (1, 8)],
                        out_features = self.model_out_features,
                        keys = self.keys,
                        batchnorm = self.batchnorm)
        
        # Criterion
        criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
        return model, criterion
    
    def A8(self): #C16C32C64D64D32D16"
        # Model
        model = VGGNet( name = 'C16C32C64D64D32D16',
                        mode = self.mode, 
                        input_num = 2, 
                        conv_layers = [(1, 16), (1, 32), (1, 64)], 
                        FC_layers = [(1, 64), (1, 32), (1, 16)],
                        out_features = self.model_out_features,
                        keys = self.keys,
                        batchnorm = self.batchnorm)
        
        # Criterion
        criterion = Loss(alpha = self.alpha, out_features = self.criterion_out_features)
        return model, criterion
    
    def A9(self): #C16C32C64C64D64D32D16
        # Model
        model = VGGNet( name = 'C16C32C64C64D64D32D16',
                        mode = self.mode, 
                        input_num = 2, 
                        conv_layers = [(1, 16), (1, 32), (1, 64), (1, 64)], 
                        FC_layers = [(1, 64), (1, 32), (1, 16)],
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
        'lr': 0.0005,  # Learning rate
        'batchsize_train': 32,
        'batchsize_val': 64,
        'max_epochs': 300,
        'max_file_num': None,
        'scheduler_stepsize': None, # 10
        'scheduler_factor': None # 0.3
    }
    
    
    
    A = A_test(mode = 0, batchnorm = True)
    train_architectures(A, data_root, ML_setting, save_folder = 'grahene_h_BN')
    
    
    
    # Test
    # model, criterion = A[0]
    # coach = Trainer(model, data_root, criterion, ML_setting)
    # coach.learn(max_epochs = 2, max_file_num = None)
    # coach.save_history('cluster/test')
    # coach.plot_history()
    