

class Data_fetch():
    """ Fetch data for analysing (not ML) """
    def __init__(self, data_root):
        
        self.data_root = data_root
        
         if isinstance(data_root, str): # Single path [string]
            indexes = self.collect_dirs(data_root, indexes)
            
        # elif  hasattr(data_root, '__len__'): # Multiple paths [list of strings]
        #     for path in data_root:
        #         indexes = self.collect_dirs(path, indexes)
                
        # else:
        #     print(f"Data root: {data_root}, not understood")
        #     exit()
        
        
    def collect_data(self, data_root:str):
        """ Go through data folder """
        
        

if __name__ == '__main__':


    