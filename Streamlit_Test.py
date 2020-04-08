import streamlit as st
import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed,Lambda, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import scipy.stats as si
import time
from scipy import stats

st.title("Deep Learning Based Pricing Test")

def calc_put_option(S, T, sigma, r):
    d1 = (np.log(S) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    Option_Price = (np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    Intrinsic_Value = np.exp(-r * T) - S
    return (-d1, -d2, Option_Price, Intrinsic_Value)

def h_asian(ST,strike,r,T):
    return([ np.exp(-r*T)*np.maximum(x-strike,0) for x in ST])

def Black_Scholes_Geometric_Exact(S, K, r, sigma1,T,n):
    sigma=sigma1*np.sqrt((n+1.0)*(2.0*n+1.0)/(6.0*n**2))
    q=(r*(n-1.0)/(2.0*n))+(sigma1**2.0)*((n+1.0)*(n-1.0)/(12.0*n**2))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (np.exp(-q * T)*S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call

def calc_d1_d2(S,K,T,sigma,r,q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    Option_Price = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return (d1,d2,Option_Price)

#'Stock Price','Strike Price','Time to Maturity','sigma','Risk free rate','Discount Factor','dividend','Dividend Factor','d1','d2'

def BS_option_accept_user_data():
    Stock_Price = st.number_input("Enter the Stock Price:",value=1.0,key="St",min_value=0.5, max_value=1.5)
    Strike_Price = st.number_input("Enter the Strike Price:",key="K",value=1.0,min_value=1.0, max_value=1.0)
    Time_to_Maturity = st.number_input("Enter the Time to Maturity:",key="T2M",value=1.0,min_value=0.01, max_value=1.00)
    Volatility = st.number_input("Enter the Volatility:",key="Vol",value=0.2,min_value=0.01, max_value=0.45)
    Risk_free_Rate = st.number_input("Enter the Risk free rate: ",key="Rf",value=0.05,min_value=0.0, max_value=0.14)
    Discount_factor = np.exp(-Risk_free_Rate*Time_to_Maturity)
    Dividend_rate=st.number_input("Enter the Dividend rate: ",key="Rf",value=0.05,min_value=0.0, max_value=0.10)
    Dividend_factor = np.exp(-Dividend_rate * Time_to_Maturity)
    d1, d2, BS_Option_Price = calc_d1_d2(Stock_Price,Strike_Price,Time_to_Maturity,Volatility,Risk_free_Rate,0.0)
    user_prediction_data = np.array([Stock_Price,Strike_Price, Time_to_Maturity,Volatility,Risk_free_Rate,Discount_factor,0.0,Dividend_factor,d1,d2]).reshape(1, -1)
    return user_prediction_data,BS_Option_Price

#S_list, K_list,T_list,sigma_list,r_list,discount_factor,q_list,dividend_factor,d1,d2

def Asian_Option_accept_user_data():
    Stock_Price = st.number_input("Enter the Stock Price:",value=1.0,key="St",min_value=0.5, max_value=1.5)
    Strike_Price = st.number_input("Enter the Strike Price:",key="K",value=1.0,min_value=1.0, max_value=1.0)
    Time_to_Maturity = st.number_input("Enter the Time to Maturity:",key="T2M",value=1.0,min_value=0.01, max_value=1.00)
    Volatility = st.number_input("Enter the Volatility:",key="Vol",value=0.2,min_value=0.01, max_value=0.45)
    Risk_free_Rate = st.number_input("Enter the Risk free rate: ",key="Rf",value=0.05,min_value=0.0, max_value=0.14)
    Geometric_Option = Black_Scholes_Geometric_Exact(Stock_Price,Strike_Price,Risk_free_Rate,Volatility,Time_to_Maturity,252)
    user_prediction_data = np.array([Stock_Price/Strike_Price, Time_to_Maturity,Volatility,Risk_free_Rate,Geometric_Option]).reshape(1, -1)
    return user_prediction_data

def american_option_accept_user_data():
    Stock_Price = st.number_input("Enter the Stock Price:",value=1.0,key="St",min_value=0.5, max_value=1.5)
    Strike_Price = st.number_input("Enter the Strike Price:",key="K",value=1.0,min_value=1.0, max_value=1.0)
    Time_to_Maturity = st.number_input("Enter the Time to Maturity:",key="T2M",value=1.0,min_value=0.01, max_value=1.00)
    Volatility = st.number_input("Enter the Volatility:",key="Vol",value=0.2,min_value=0.01, max_value=0.45)
    Risk_free_Rate = st.number_input("Enter the Risk free rate: ",key="Rf",value=0.05,min_value=0.0, max_value=0.14)
    d1, d2, BS_Option_Price, Intrinsic_Value = calc_put_option(Stock_Price/Strike_Price,Time_to_Maturity,Volatility,Risk_free_Rate)
    user_prediction_data = np.array([Stock_Price/Strike_Price, Time_to_Maturity,Volatility,Risk_free_Rate,d1,d2,BS_Option_Price,Intrinsic_Value]).reshape(1, -1)
    return user_prediction_data

def h(ST, strike, r, T):
    payoff = np.exp(-r * T) * np.maximum(strike - ST, 0)
    return (payoff)

    #"Stock_Price", "Time To Maturity", "Volatility", "Risk Free Rate",d1,d2,BS_Option_Price,Intrinsic_Value

def compile_model(input_shape):
    model = Sequential()
    model.add(Dense(400, input_dim=input_shape, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(1, activation='relu'))
    adam =Adam(lr=0.005,decay=1e-6)
    model.compile(loss='mape', optimizer= adam, metrics=['mape'])
    return(model)

def Longstaff_Schwartz(S0, K, r, q, sigma, T, N, n):
    Average_Black_Scholes_price = Put_Option(S0, T, sigma, r)
    z = np.random.standard_normal((n, N))
    z1 = -z
    Simulated_Prices = np.zeros((n, N))
    Simulated_Prices_Antithetic = np.zeros((n, N))
    delta_t = T / N
    f1 = (r - q - 0.5 * sigma ** 2) * delta_t
    f2 = sigma * np.sqrt(delta_t)
    Simulated_Prices[:, 0] = S0 * np.exp(f1 + f2 * z[:, 0])
    Simulated_Prices_Antithetic[:, 0] = S0 * np.exp(f1 + f2 * z1[:, 0])
    for j in range(1, N):
        Simulated_Prices[:, j] = Simulated_Prices[:, j - 1] * np.exp(f1 + f2 * z[:, j])
        Simulated_Prices_Antithetic[:, j] = Simulated_Prices_Antithetic[:, j - 1] * np.exp(f1 + f2 * z1[:, j])
    Simulated_Prices = np.concatenate((np.ones((n, 1)) * S0, Simulated_Prices), axis=1)
    Simulated_Prices_Antithetic = np.concatenate((np.ones((n, 1)) * S0, Simulated_Prices_Antithetic), axis=1)
    Final_Prices = Simulated_Prices[:, N - 1]
    Final_Prices_Antithetic = Simulated_Prices_Antithetic[:, N - 1]
    Average_Payoff = h(Final_Prices, K, r, T)
    Average_Payoff_Antithetic = h(Final_Prices_Antithetic, K, r, T)
    S = np.concatenate((Simulated_Prices, Simulated_Prices_Antithetic), axis=0)
    time_points = np.linspace(0, T, num=N + 1, endpoint=True)
    Cash_Flow = np.zeros((2 * n, N + 1))
    Cash_Flow[:, N] = np.maximum(0, K - S[:, N])
    for j in range(N - 1, -1, -1):
        payoff_vector = np.maximum(0, K - S[:, j])
        Y = np.sum(np.exp(-r * (time_points[j + 1:] - time_points[j])) * Cash_Flow[:, j + 1:], axis=1)
        continuation_vector = Regression_Function(S[:, j], K, Y)
        Cash_Flow[:, j] = (payoff_vector >= np.array(continuation_vector)) * payoff_vector
        Cash_Flow[Cash_Flow[:, j] > 0, j + 1:] = 0
    Cash_Flow_1 = Cash_Flow[0:n:1, :]
    Cash_Flow_2 = Cash_Flow[n:2 * n:1, :]
    Cash_Flow_1 = np.exp(-r * time_points) * Cash_Flow_1
    Cash_Flow_2 = np.exp(-r * time_points) * Cash_Flow_2
    price1 = (np.sum((Cash_Flow_1), axis=1))
    price2 = (np.sum((Cash_Flow_2), axis=1))
    Corr_coef = np.corrcoef(price1, Average_Payoff)[0, 1]
    if (np.std(Average_Payoff) == 0.0):
        alpha = 0.0
    else:
        alpha = -Corr_coef * (np.std(price1) / np.std(Average_Payoff))
    price1 = price1 + alpha * (Average_Payoff - Average_Black_Scholes_price)
    Corr_coef = np.corrcoef(price2, Average_Payoff_Antithetic)[0, 1]
    if (np.std(Average_Payoff_Antithetic) == 0.0):
        alpha = 0.0
    else:
        alpha = -Corr_coef * (np.std(price1) / np.std(Average_Payoff_Antithetic))
    price2 = price2 + alpha * (Average_Payoff_Antithetic - Average_Black_Scholes_price)
    price = (price2 + price1) / 2
    return (np.mean(price), np.std(price, ddof=1) / np.sqrt(n))

def Regression_Function(St, K, DCF):
    L0 = np.exp(-St / (K * 2.0))
    L1 = np.exp(-St / (K * 2.0)) * (1.0 - (St / K))
    L2 = np.exp(-St / (K * 2.0)) * (1.0 - 2.0 * (St / K) + ((St / K) ** 2 / 2.0))
    A = np.vstack([np.ones(len(St)), L0, L1, L2]).T
    A[np.where(St > K), :] = 0.0
    B0, B1, B2, B3 = np.linalg.lstsq(A, DCF, rcond=None)[0]
    Continuation_Value = B0 + B1 * L0 + B2 * L1 + B3 * L2
    Continuation_Value[np.where(St > K)] = 0
    return Continuation_Value

    #"Stock_Price", "Time To Maturity", "Volatility", "Risk Free Rate",d1,d2,BS_Option_Price,Intrinsic_Value
    #Longstaff_Schwartz(S0, K, r, q, sigma, T, N, n):


def MC_AO_AN_CV_ST(S0, K, r, sigma, T, n, Numer_of_Periods):
    Average_Geometric_Price = Black_Scholes_Geometric_Exact(S0, K, r, sigma, T, Numer_of_Periods)
    delta_T = 1.0 / Numer_of_Periods
    Simulated_Prices = np.zeros((Numer_of_Periods, n))
    Simulated_Prices_A = np.zeros((Numer_of_Periods, n))
    Average_Prices = np.zeros(n)
    Average_Prices_A = np.zeros(n)
    Average_Payoff = np.zeros(n)
    Average_Payoff_A = np.zeros(n)
    f1 = (r - 0.5 * sigma ** 2) * delta_T
    f2 = sigma * np.sqrt(delta_T)
    z = np.random.normal(size=(Numer_of_Periods, n))
    z_A = -z
    Simulated_Prices[:1, ] = S0 * np.exp(f1 + f2 * z[1, :])
    Simulated_Prices_A[:1, ] = S0 * np.exp(f1 + f2 * z_A[1, :])
    for j in range(Numer_of_Periods - 1):
        Simulated_Prices[j + 1, :] = Simulated_Prices[j, :] * np.exp(f1 + f2 * z[(j + 1), :])
        Simulated_Prices_A[j + 1, :] = Simulated_Prices_A[j, :] * np.exp(f1 + f2 * z_A[(j + 1), :])
    Average_Prices = Simulated_Prices.mean(0)
    Average_Prices_A = Simulated_Prices_A.mean(0)
    Average_Payoff = h_asian(Average_Prices, K, r, T)
    Average_Payoff_A = h_asian(Average_Prices_A, K, r, T)
    Geometric_mean = stats.gmean(Simulated_Prices, axis=0)
    Geometric_Option_Prices = h_asian(Geometric_mean, K, r, T)
    Geometric_mean_A = stats.gmean(Simulated_Prices_A, axis=0)
    Geometric_Option_Prices_A = h_asian(Geometric_mean_A, K, r, T)
    Corr_coef = np.corrcoef(Geometric_Option_Prices, Average_Payoff)[0, 1]
    if (np.std(Geometric_Option_Prices) == 0.0):
        alpha = 0.0
    else:
        alpha = -Corr_coef * (np.std(Average_Payoff) / np.std(Geometric_Option_Prices))
    Average_Payoff = Average_Payoff + alpha * (Geometric_Option_Prices - Average_Geometric_Price)
    print(alpha)
    Corr_coef_A = np.corrcoef(Geometric_Option_Prices_A, Average_Payoff_A)[0, 1]
    if (np.std(Geometric_Option_Prices_A) == 0.0):
        alpha_A = 0.0
    else:
        alpha_A = -Corr_coef_A * (np.std(Average_Payoff_A) / np.std(Geometric_Option_Prices_A))
    Average_Payoff_A = Average_Payoff_A + alpha_A * (Geometric_Option_Prices_A - Average_Geometric_Price)
    Average_Payoff = np.mean([Average_Payoff, Average_Payoff_A], axis=0)
    return (np.mean(Average_Payoff), np.std(Average_Payoff)/np.sqrt(n))

def Put_Option(S, T, sigma, r):
    d1 = (np.log(S) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    Option_Price = (np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    return (np.array(Option_Price))

value = st.sidebar.selectbox("Select Type of Option", ["European Call Option","American Put Option", "Asian Call Option"])

if (value == "European Call Option"):

    x_train,BS_Option_Price=BS_option_accept_user_data()
    if st.button('Price European Call Option'):

        # PRICING USING MONTE CARLO
        start_1 = time.time()
        dummy = Put_Option(x_train[0,0], x_train[0,1], x_train[0,2], x_train[0,3])
        stop_1 = time.time()
        MC_Option_Price = round(float(BS_Option_Price), 6)
        MC_Time_Taken = round(float(stop_1 - start_1) * 1000, 0)
        MC_Stderr = 0.0
        students = [(BS_Option_Price, MC_Time_Taken, MC_Stderr)]

        #PRICING USING NEURAL NETWORK
        model_BS=compile_model(x_train.shape[1])
        #model_BS.load_weights('/home/abhishek/Desktop/Black Scholes/Python Files/file_output_relu400_mse_final.hdf5')
        model_BS.load_weights('European_Option_Final_Weights.hdf5')
        start = time.time()
        prediction = model_BS.predict(x_train)
        stop = time.time()
        NN_Option_Price = np.maximum(round(float(prediction - 1), 6),0)
        NN_Time_Taken = round(float(stop - start) * 1000, 0)
        NN_Error = int(((MC_Option_Price - NN_Option_Price)*100000))
        students = [(BS_Option_Price, MC_Time_Taken, MC_Stderr), (NN_Option_Price, NN_Time_Taken, NN_Error)]
        dfObj = pd.DataFrame(students, columns=['Option Price', 'Time taken in Miliseconds', 'Pricing Error per mil$'],
                             index=['Closed Form', 'Neural Network'])
        st.table(dfObj)

elif(value == "American Put Option"):

    x_train = american_option_accept_user_data()
    test_size = st.number_input("Enter Number of trials:", key="trials", value=1)
    if st.button('Price American Put Option'):

        # PRICING USING MONTE CARLO
        start_1 = time.time()
        American_Option_Price, American_Option_Price_Stderr = Longstaff_Schwartz(x_train[0,0], 1, x_train[0,3], 0, x_train[0,2], x_train[0,1], 52, 50000)
        stop_1 = time.time()
        MC_Option_Price = round(float(American_Option_Price), 6)
        MC_Time_Taken = round(float(stop_1 - start_1) * 1000, 0)
        MC_Stderr = int(round((American_Option_Price_Stderr), 8)*200000)
        students = [(American_Option_Price, MC_Time_Taken, MC_Stderr)]

        #PRICING USING NEURAL NETWORK
        model_american=compile_model(x_train.shape[1])
        #model_american.load_weights('/home/abhishek/Desktop/American Options/file_output_relu400_mse_american_t2m.hdf5')
        model_american.load_weights('American_Option_Final_Weights.hdf5')
        start = time.time()
        prediction = model_american.predict(x_train)
        stop = time.time()
        NN_Option_Price = np.maximum(round(float(prediction - 1), 6),0)
        NN_Time_Taken = round(float(stop - start) * 1000, 0)
        NN_Error = int(((MC_Option_Price - NN_Option_Price)*100000))
        students = [(American_Option_Price, MC_Time_Taken, MC_Stderr), (NN_Option_Price, NN_Time_Taken, NN_Error)]
        dfObj = pd.DataFrame(students, columns=['Option Price', 'Time taken in Miliseconds', 'Pricing Error per mil$'],
                             index=['Monte Carlo', 'Neural Network'])
        st.table(dfObj)

else:

    x_train = Asian_Option_accept_user_data()
    test_size = st.number_input("Enter Number of trials:", key="trials", value=1)
    if st.button('Price Asian Call Option'):
        # Stock_Price/Strike_Price, Time_to_Maturity,Volatility,Risk_free_Rate,Geometric_Option
        # PRICING USING MONTE CARLO S0, K, r, sigma, T, n, Numer_of_Periods
        start_1 = time.time()
        Asian_Option_Price, Asian_Option_Price_Stderr = MC_AO_AN_CV_ST(x_train[0,0], 1.0,x_train[0,3],x_train[0,2],x_train[0,1],5000,252)
        stop_1 = time.time()
        MC_Option_Price = round(float(Asian_Option_Price), 6)
        MC_Time_Taken = round(float(stop_1 - start_1) * 1000, 0)
        MC_Stderr = int(round((Asian_Option_Price_Stderr), 8)*200000)
        students = [(Asian_Option_Price, MC_Time_Taken, MC_Stderr)]

        #PRICING USING NEURAL NETWORK
        model_Asian = compile_model(x_train.shape[1])
        #model_Asian.load_weights('/home/abhishek/Desktop/Asian Option/Weights/file_output_relu400_mape_Asian.hdf5')
        model_Asian.load_weights('Asian_Option_Final_Weights.hdf5')
        start = time.time()
        prediction = model_Asian.predict(x_train)
        stop = time.time()
        NN_Option_Price = np.maximum(round(float(prediction - 1), 6),0)
        NN_Time_Taken = round(float(stop - start) * 1000, 0)
        NN_Error = int(((MC_Option_Price - NN_Option_Price)*100000))
        students = [(Asian_Option_Price, MC_Time_Taken, MC_Stderr), (NN_Option_Price, NN_Time_Taken, NN_Error)]
        dfObj = pd.DataFrame(students, columns=['Option Price', 'Time taken in Miliseconds', 'Pricing Error per mil$'],
                             index=['Monte Carlo', 'Neural Network'])
        st.table(dfObj)


