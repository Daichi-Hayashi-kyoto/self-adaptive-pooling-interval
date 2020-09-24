import scipy.stats as st
import time
mu, sigma = st.norm.fit(log_score, loc = -7, scale = 0.5)
#rv = st.norm(loc = mu, scale = sigma)

#a_th_1st = rv.ppf(q=0.5)
#a_th = rv.ppf(q = 0.95)
"""
前処理コードは別途必要.
"""


def my_Algo1(window_size, data, Polling_time, f_min, f_max, past_anomaly_score, mu, sigma,  T_old = 10):

    '''
    dataはnumpy配列を想定
    past_anomaly_score は list型
    '''

    X = data
    N = window_size
    T = [Polling_time[i] for i in range(len(Polling_time))]
    window_size_list = []
    window_size_list.append(N)  
    train_size = int(np.round(X.shape[0] * 0.15))
    sample_queue = list(X[:train_size])
    test_data = X[train_size:]

    T.append(T_old)
    refer_index = (train_size - 1) + int(np.round(T[-1]/5))   # 参照するindex値
    Polling_time_list = []
    Polling_time_list.append(T[-1])
    test_index = int(np.round(T[-1]/5)) - 1 # 参照するtestデータのindex値
    refer_index_list = []
    predict_list = []
    test_index_list = []
    test_index_list.append(test_index)
    count = 1
    anomaly_score_list = []

    while refer_index <= X.shape[0]:

        '''
        次のデータ点の予測
        '''
        start_time = time.time()
        sum_N = sum(steep_rate[-(N-1):])
        x_pred = sample_queue[-1] + (T[-1]/(N-1)) * sum_N
        

        print("=======================================================")
        print("sample_queue [-1] は{}です。".format(sample_queue[-1]))
        print("(T[-1]/(N-1)) * sum_Nは{}です。".format((T[-1]/(N-1)) * sum_N))
        print("現在参照しているデータのindex値は{}です。".format(refer_index))
        print("Nの値は{}です".format(N))

        

        # 実測値を受けとる
        if test_index >= test_data.shape[0]:
            break
            
        predict_list.append(x_pred)
        refer_index_list.append(refer_index)
        x_real = test_data[test_index]    


        '''
        類似度ベクトルの計算
        '''

        slope_predict = np.array([x_pred - sample_queue[-1], T[-1]]).reshape(-1, 1)
        slope_real = np.array([x_real - sample_queue[-1], T[-1] ]).reshape(-1, 1)

        similar_vec_score = np.abs(np.dot(slope_predict.T, slope_real)/(np.linalg.norm(slope_predict) * np.linalg.norm(slope_real)))
        anomaly_score = 1 - similar_vec_score
        if not isinstance(anomaly_score, np.ndarray):
            anomaly_score = np.array(anomaly_score)
            
            
        anomaly_score_list.append(anomaly_score[0][0])
        past_anomaly_score.append(np.log(anomaly_score[0][0]))
        
        #print("実測値と予測値の誤差は{}です".format(x_pred - x_real ))
        #print("異常スコアは{}です。".format(anomaly_score[0][0]))


        """
        window size 内のデータで正規分布のフィッティング

        """

        past_anomaly_list = past_anomaly_score[-N:]     # 過去N個のデータ点を参照して推定に用いるデータに代入
        #print(past_anomaly_list)
        mu, sigma = st.norm.fit(past_anomaly_list, loc = mu, scale = sigma)   # fitting
        rv = st.norm(loc = mu, scale = sigma)
        a_th_1st = rv.ppf(q = 0.8)
        a_th = rv.ppf(q = 0.97)
        print("第一閾値は{}".format(a_th_1st))
        print("第二閾値は{}".format(a_th))




        '''
        T の更新部分
        '''

        log_anomaly_score = np.log(anomaly_score[0][0])
        print("対数異常度は{}".format(log_anomaly_score))
        if log_anomaly_score <= a_th_1st:
            T_new = 1/f_min

        elif a_th_1st <= log_anomaly_score <= a_th:

            T_new = (1/f_min - 1/f_max)/(a_th_1st - a_th) * (log_anomaly_score - a_th) + 1/f_max

        else:
            T_new = 1/f_max


        sample_queue.append(x_real)
        
        print("次回のPolling timeは{}秒後です".format(T_new))
        T.append(T_new)


        '''
        AIMDの実装部分
        '''
        if T_new >= T_old:
            N += 1

        else:
            N = int(np.round(N * 0.8))
            if N<5:
                N=5

        window_size_list.append(N)
        refer_index += int(np.round(T[-1]/5))
        test_index += int(np.round(T[-1]/5))
        if test_index < len(test_data):
            test_index_list.append(test_index)
            
        Polling_time_list.append(T_new)
        steep_rate.append((sample_queue[-1] - sample_queue[-2])/T_old)
        print("steep_lateに{}を追加しました".format((sample_queue[-1] - sample_queue[-2])/T_old))
        print("次回のPolling timeは{}秒後です".format(T_new))
        print("次の参照データのindex値は{}です。".format(refer_index))
        print("{}回目終了".format(count))
        T_old = T_new
        end_time = time.time()
        count += 1
        print("１回の処理時間は{}秒です".format(end_time - start_time))

    return Polling_time_list, window_size_list, refer_index_list, predict_list, anomaly_score_list, test_index_list, count


if name == "__main__":
    polling_time_list, window_size_list, refer_index_list, predict_list, anomaly_score_list, test_index, count = my_Algo1(
    
                                                                window_size = 100, 
                                                                data = X,
                                                                Polling_time = polling_time, 
                                                                f_max = 0.2, f_min = 1/100,
                                                                past_anomaly_score = log_score_list,
                                                                mu = mu,
                                                                sigma = sigma,                            
                                                                T_old = 30
                                                                )

