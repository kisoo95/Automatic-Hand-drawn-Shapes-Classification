import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import ImageFont, ImageDraw, Image
from scipy.stats import gaussian_kde
import scipy.stats as stats

# 적합한 회전 각도 찾기
class ExtractFromBorderedDocument():
    def __init__(self, cv2_candidate, how_much_range = 50, sig__ = 20, normal_ran = 5, extract_max_num = 3
                 , sigmoid_middle = 0.05, sigmoid_weight = 100, horizontal_partition = 3, vertical_partition = 4):
        self.cv2_image = cv2_candidate
        self.extract_max_num = extract_max_num
        self.how_much_range = how_much_range # how_much_range는 정규분포의 평균을 정하기 위해 중심으로부터 확인 범위
        self.sigma, self.normal_range = sig__, normal_ran # sigma, normal_range는 각각 정규분포 표준편차, plot에 나타내는 범위
        self.sigmoid_middle, self.sigmoid_weight = sigmoid_middle, sigmoid_weight
        self.how_much_0, self.how_much_1 = horizontal_partition, vertical_partition # how_much_0, how_much_1는 각각 x축을 몇개로 나눌것인지, y축을 몇개로 나눌것인지
        
    # 일부분에 대한 pdf, 해당하는 normal distribution 그리는 함수
    def calculate_pdf_normal(self, partition_idx, axis_num, how_much__, pdf_plot = False, normal_plot = False, rotation_status = False):
        
        # 시그모이드 함수 투여
        if rotation_status == True:
            squared_c = self.squared_color
        else:
            squared_c = 1/(1 + np.exp((-self.sigmoid_weight)*(self.cv2_image/np.max(self.cv2_image) - self.sigmoid_middle)))
        
        axis__len = len(np.mean(squared_c, axis=axis_num))
        
        if partition_idx == (how_much__ - 1):
            # 기존 pdf
            np_list = np.mean(squared_c, axis=axis_num)[int(partition_idx*axis__len/how_much__):]
            range_list = list(range(int(partition_idx*axis__len/how_much__),axis__len))
            if pdf_plot:
                plt.plot(range_list,np_list/np.sum(np_list))
                
            # 정규분포 plot (mu = 확인 범위 내에서 가장 최대값인 곳)
            temp_np_list = np_list[len(np_list)//2-self.how_much_range:len(np_list)//2+self.how_much_range]
            list_of_maxima = np.where(np.max(temp_np_list) == np_list)
            where_is_max = int(np.mean(list_of_maxima)) # 간혹 최대치가 중복되어 여러개로 나올 수 있음 -> 평균화
            mu = range_list[where_is_max]
            if normal_plot:
                x = np.linspace(mu - self.normal_range*self.sigma, mu + self.normal_range*self.sigma, 200)
                final_stats_norm_pdf = stats.norm.pdf(x, mu, self.sigma)
                plt.plot(x, final_stats_norm_pdf/max(final_stats_norm_pdf))
                
            # 최대치 저장
            max_point_, max_pdf_list = mu,sorted(temp_np_list, reverse=True)[:self.extract_max_num]
        else:
            # 기존 plot
            np_list = np.mean(squared_c, axis=axis_num)[int(partition_idx*axis__len/how_much__):int((partition_idx+1)*axis__len/how_much__)]
            range_list = list(range(int(partition_idx*axis__len/how_much__),int((partition_idx+1)*axis__len/how_much__)))
            if pdf_plot:
                plt.plot(range_list, np_list/np.sum(np_list))
            
            # 정규분포 plot (mu = 확인 범위 내에서 가장 최대값인 곳)
            temp_np_list = np_list[len(np_list)//2-self.how_much_range:len(np_list)//2+self.how_much_range]
            list_of_maxima = np.where(np.max(temp_np_list) == np_list)
            where_is_max = int(np.mean(list_of_maxima)) # 간혹 최대치가 중복되어 여러개로 나올 수 있음 -> 평균화
            mu = range_list[where_is_max]
            if normal_plot:
                x = np.linspace(mu - self.normal_range*self.sigma, mu + self.normal_range*self.sigma, 200)
                final_stats_norm_pdf = stats.norm.pdf(x, mu, self.sigma)
                plt.plot(x, final_stats_norm_pdf/max(final_stats_norm_pdf))
            
            # 최대치 저장
            max_point_, max_pdf_list = mu, sorted(temp_np_list, reverse=True)[:self.extract_max_num]
        
        return max_point_, max_pdf_list
    
    # 적합한 중심점 탐색 및 중심 부분 pdf 계산
    def calculate_mu_pdf_at_specific_angle(self, theta_, draw_plot = False, draw_norm = False, plt_num = 241):
        (h, w) = self.cv2_image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), theta_, 1.0)
        cv2_image__ = cv2.warpAffine(self.cv2_image, M, (w, h))    
        
        # 시그모이드 함수 투여
        self.squared_color = 1/(1 + np.exp((-self.sigmoid_weight)*(cv2_image__/np.max(cv2_image__) - self.sigmoid_middle)))
        
        if draw_plot:
            plt.figure(figsize = (11,5))
            
        # plt_num는 plot 숫자  
        np_1_mu, np_1_pdf, np_0_mu, np_0_pdf = [], [], [], []
        for jj in range(self.how_much_1):
            if draw_plot:
                plt.subplot(plt_num)
                plt_num += 1
            # 클래스 내 함수를 가져오기 위해선 이렇게 입력해야 하기
            wow_1, wow_2 = self.calculate_pdf_normal(partition_idx = jj, axis_num = 1, how_much__ = self.how_much_1
                                          , pdf_plot = draw_plot, normal_plot = draw_norm, rotation_status = True)
            np_1_mu.append(wow_1)
            np_1_pdf.append(wow_2)
        
        np_1_mu, np_1_pdf = np.array(np_1_mu), np.array(np_1_pdf)
        
        for jj in range(self.how_much_0):
            if draw_plot:
                plt.subplot(plt_num)
                plt_num += 1
            # 클래스 내 함수를 가져오기 위해선 이렇게 입력해야 하기
            wow_1, wow_2 = self.calculate_pdf_normal(partition_idx = jj, axis_num = 0, how_much__ = self.how_much_0
                                          , pdf_plot = draw_plot, normal_plot = draw_norm, rotation_status = True)
            np_0_mu.append(wow_1)
            np_0_pdf.append(wow_2)
            
        np_0_mu, np_0_pdf = np.array(np_0_mu), np.array(np_0_pdf)
        
        return np_1_mu, np_1_pdf, np_0_mu, np_0_pdf
    
    # 적합한 회전 각도 찾기
    def find_the_angle(self, start_theta = -2, finish_theta = 2, bin_theta = 100, d_plot = False, d_norm = False, plot_num = 241):       
        true_max_theta, true_max_pdf = 0, 0
        
        for theta in range(start_theta*bin_theta, finish_theta*bin_theta):
            # 클래스 내 함수를 가져오기 위해선 이렇게 입력해야 하기
            return_box = self.calculate_mu_pdf_at_specific_angle(theta_ = theta/bin_theta
                                                    , draw_plot = d_plot, draw_norm = d_norm, plt_num = plot_num)
            test_max_pdf = np.median(return_box[1]) + np.median(return_box[3])
            if true_max_pdf < test_max_pdf:
                true_max_theta, true_max_pdf, true_return_box = theta/bin_theta, test_max_pdf, return_box
        
        return true_max_theta

