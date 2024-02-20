import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import ImageFont, ImageDraw, Image
from scipy.stats import gaussian_kde
import scipy.stats as stats

# 일부분에 대한 pdf, 해당하는 normal distribution 그리는 함수
def pdf_and_normal(squared_c, j, axis_num, how_much__, how_much_range, normal_range, sigma, extract_max_num = 3, pdf_plot = False, normal_plot = False):
    axis__len = len(np.mean(squared_c, axis=axis_num))
    
    if j == (how_much__ - 1):
        # 기존 pdf
        np_list = np.mean(squared_c, axis=axis_num)[int(j*axis__len/how_much__):]
        range_list = list(range(int(j*axis__len/how_much__),axis__len))
        if pdf_plot:
            plt.plot(range_list,np_list/np.sum(np_list))
            
        # 정규분포 plot (mu = 확인 범위 내에서 가장 최대값인 곳)
        temp_np_list = np_list[len(np_list)//2-how_much_range:len(np_list)//2+how_much_range]
        list_of_maxima = np.where(np.max(temp_np_list) == np_list)
        where_is_max = int(np.mean(list_of_maxima)) # 간혹 최대치가 중복되어 여러개로 나올 수 있음 -> 평균화
        mu = range_list[where_is_max]
        if normal_plot:
            x = np.linspace(mu - normal_range*sigma, mu + normal_range*sigma, 200)
            final_stats_norm_pdf = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, final_stats_norm_pdf/max(final_stats_norm_pdf))
            
        # 최대치 저장
        max_point_, max_pdf_list = mu,sorted(temp_np_list, reverse=True)[:extract_max_num]
    else:
        # 기존 plot
        np_list = np.mean(squared_c, axis=axis_num)[int(j*axis__len/how_much__):int((j+1)*axis__len/how_much__)]
        range_list = list(range(int(j*axis__len/how_much__),int((j+1)*axis__len/how_much__)))
        if pdf_plot:
            plt.plot(range_list, np_list/np.sum(np_list))
        
        # 정규분포 plot (mu = 확인 범위 내에서 가장 최대값인 곳)
        temp_np_list = np_list[len(np_list)//2-how_much_range:len(np_list)//2+how_much_range]
        list_of_maxima = np.where(np.max(temp_np_list) == np_list)
        where_is_max = int(np.mean(list_of_maxima)) # 간혹 최대치가 중복되어 여러개로 나올 수 있음 -> 평균화
        mu = range_list[where_is_max]
        if normal_plot:
            x = np.linspace(mu - normal_range*sigma, mu + normal_range*sigma, 200)
            final_stats_norm_pdf = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, final_stats_norm_pdf/max(final_stats_norm_pdf))
        
        # 최대치 저장
        max_point_, max_pdf_list = mu, sorted(temp_np_list, reverse=True)[:extract_max_num]
    
    return max_point_, max_pdf_list

# 적합한 중심점 탐색 및 중심 부분 pdf 계산
def finding_the_box(cv2_images, theta_, bin_theta_, draw_plot = False, draw_norm = False, sigmoid_middle = 0.05, sigmoid_weight = 100
                    , plt_num = 241, how_much_0 = 3, how_much_1 = 4, how_much_range_from_mean = 50, sig_ = 20, normal_ran = 5
                    , extracting_max_num = 3):
    
    (h, w) = cv2_images.shape[:2]
    (cX, cY) = (w // 2, h // 2)
        
    M = cv2.getRotationMatrix2D((cX, cY), theta/bin_theta, 1.0)
    cv2_images = cv2.warpAffine(cv2_images, M, (w, h))    
    
    # 시그모이드 함수 투여
    squared_color = 1/(1 + np.exp((-sigmoid_weight)*(cv2_images/np.max(cv2_images) - sigmoid_middle)))
    
    if draw_plot:
        plt.figure(figsize = (11,5))
        
    # plt_num, how_much_0, how_much_1는 각각 plot 숫자, x축을 몇개로 나눌것인지, y축을 몇개로 나눌것인지    
    # how_much_range_from_mean는 정규분포의 평균을 정하기 위해 중심으로부터 확인 범위
    # sig_, normal_ran는 각각 정규분포 표준편차, plot에 나타내는 범위
    np_1_mu, np_1_pdf, np_0_mu, np_0_pdf = [], [], [], []
    for jj in range(how_much_1):
        if draw_plot:
            plt.subplot(plt_num)
            plt_num += 1
        wow_1, wow_2 = pdf_and_normal(squared_color, jj, axis_num = 1, how_much__ = how_much_1, how_much_range = how_much_range_from_mean
                       , normal_range = normal_ran, sigma = sig_, extract_max_num = extracting_max_num
                              , pdf_plot = draw_plot, normal_plot = draw_norm)
        np_1_mu.append(wow_1)
        np_1_pdf.append(wow_2)
    
    np_1_mu, np_1_pdf = np.array(np_1_mu), np.array(np_1_pdf)
    
    for jj in range(how_much_0):
        if draw_plot:
            plt.subplot(plt_num)
            plt_num += 1
        wow_1, wow_2 = pdf_and_normal(squared_color, jj, axis_num = 0, how_much__ = how_much_0, how_much_range = how_much_range_from_mean
                       , normal_range = normal_ran, sigma = sig_, extract_max_num = extracting_max_num
                              , pdf_plot = draw_plot, normal_plot = draw_norm)
        np_0_mu.append(wow_1)
        np_0_pdf.append(wow_2)
        
    np_0_mu, np_0_pdf = np.array(np_0_mu), np.array(np_0_pdf)
    
    return np_1_mu, np_1_pdf, np_0_mu, np_0_pdf

