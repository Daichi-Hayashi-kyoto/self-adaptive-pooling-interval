import numpy as np


class rmse:

    def __init__(self, data, test_data, refer_index, test_index, sample_queue):
        '''
        data はnumpy 配列を仮定
        test_index : テストデータに対するインデックス値. 全体を通してでのindex値ではないことに注意
        '''

        self.data = data
        self.refer_index = refer_index     # データ全体を通してのindex値
        self.test_index = test_index
        self.test_data = test_data
        self.sample_queue = sample_queue      # 実際にアルゴリズム中にサンプリングしたもの．アルゴリズム中のsample_queueではないことに注意
        self.N =len(data)     # original データの数
        self.train_size = int(np.round(self.N * 0.15))
        self.polling_times = len(test_index)
        self.test_number = len(test_data)
        #self.embed_list = np.zeros(self.test_number)  


    def make_list(self):

        self.embed_list = np.zeros(self.test_number)        # rmseを評価するための箱

        for i in self.test_index:
            #print("入るテスト値は{}".format(self.test_data[i]))
            self.embed_list[i] = self.test_data[i]
            
        

        for k in range(1, len(self.sample_queue)):
            self.x_1, self.x_2 = self.sample_queue[k-1], self.sample_queue[k]
            #print("現在参照している参照値は{}と{}です。".format(self.refer_index[k-1], self.refer_index[k]))
            self.linear_interpolation(self.refer_index[k-1], self.refer_index[k])

        return self.embed_list
            
    
    def evaluation(self):
        self.embed_list = self.make_list()
        
        for i in range(len(self.embed_list)):
            if self.embed_list[i] == 0:
                self.embed_list[i] = self.test_data[i]

        return np.sqrt(np.mean((self.embed_list - self.test_data)**2))


    def linear_interpolation(self, index_1, index_2):
        '''
        線形補間するところ
        '''

        if index_1 < index_2: 
            for j in range(index_2 - index_1 - 1):
                f = (self.x_1 - self.x_2)/(5 * (index_2 - index_1 + 1) ) * 5 * (j + 1) + self.x_1
            
                self.embed_list[(index_1 - self.train_size) + j  + 1] = f     # 線形補間


        else:
            print("index_2 must be more than index_1")