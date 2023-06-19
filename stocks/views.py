from django.shortcuts import render
import pandas as pd
from .model import get_historical


from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
# import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
# from Tweet import Tweet
import nltk
import os

os.environ["NLTK_DATA"] = "https://nltk.org/nltk_data/"
nltk.download('punkt')

def home(request):
    return render(request, "index.html")


def stock_page(request):
    stock_name = request.GET.get('stock_name')
    quote = stock_name
    get_historical(quote)
    df = pd.read_csv(''+quote+'.csv')
    print("##############################################################################")
    print("Today's",quote,"Stock Data: ")
    today_stock = df.iloc[-1:]
    print(today_stock)
    print("##############################################################################")
    df = df.dropna()
    code_list = [quote] * len(df)
    df2 = pd.DataFrame(code_list, columns=['Code'])
    df2 = pd.concat([df2, df], axis=1)
    df = df2

    def ARIMA_ALGO(df):
        uniqueVals = df["Code"].unique()  
        len(uniqueVals)
        df=df.set_index("Code")
        #for daily basis
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')
        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1 ,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            return predictions
        for company in uniqueVals[:10]:
            data=(df.loc[company,:]).reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Price','Date']]
            Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'],axis =1)
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(Quantity_date)
            plt.savefig('static/Trends.png')
            plt.close(fig)
            
            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]
            #fit in model
            predictions = arima_model(train, test)
            
            #plot graph
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(test,label='Actual Price')
            plt.plot(predictions,label='Predicted Price')
            plt.legend(loc=4)
            # plt.savefig('static/ARIMA.png')
            # plt.close(fig)
            print()
            print("##############################################################################")
            arima_pred=predictions[-2]
            print("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
            #rmse calculation
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            print("ARIMA RMSE:",error_arima)
            print("##############################################################################")
            return arima_pred, error_arima
        
    def LSTM_ALGO(df):
            #Split data into training set and test set
            dataset_train=df.iloc[0:int(0.8*len(df)),:]
            dataset_test=df.iloc[int(0.8*len(df)):,:]
            ############# NOTE #################
            #TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
            # HERE N=7
            ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
            training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
            #select cols using above manner to select as float64 type, view in var explorer

            #Feature Scaling
            from sklearn.preprocessing import MinMaxScaler
            sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
            training_set_scaled=sc.fit_transform(training_set)
            #In scaling, fit_transform for training, transform for test
            
            #Creating data stucture with 7 timesteps and 1 output. 
            #7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
            X_train=[]#memory with 7 days from day i
            y_train=[]#day i
            for i in range(7,len(training_set_scaled)):
                X_train.append(training_set_scaled[i-7:i,0])
                y_train.append(training_set_scaled[i,0])
            #Convert list to numpy arrays
            X_train=np.array(X_train)
            y_train=np.array(y_train)
            X_forecast=np.array(X_train[-1,1:])
            X_forecast=np.append(X_forecast,y_train[-1])
            #Reshaping: Adding 3rd dimension
            X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
            X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
            #For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)
            
            #Building RNN
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.layers import Dropout
            from keras.layers import LSTM
            
            #Initialise RNN
            regressor=Sequential()
            
            #Add first LSTM layer
            regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
            #units=no. of neurons in layer
            #input_shape=(timesteps,no. of cols/features)
            #return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
            regressor.add(Dropout(0.1))
            
            #Add 2nd LSTM layer
            regressor.add(LSTM(units=50,return_sequences=True))
            regressor.add(Dropout(0.1))
            
            #Add 3rd LSTM layer
            regressor.add(LSTM(units=50,return_sequences=True))
            regressor.add(Dropout(0.1))
            
            #Add 4th LSTM layer
            regressor.add(LSTM(units=50))
            regressor.add(Dropout(0.1))
            
            #Add o/p layer
            regressor.add(Dense(units=1))
            
            #Compile
            regressor.compile(optimizer='adam',loss='mean_squared_error')
            
            #Training
            regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
            #For lstm, batch_size=power of 2
            
            #Testing
            ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
            real_stock_price=dataset_test.iloc[:,4:5].values
            
            #To predict, we need stock prices of 7 days before the test set
            #So combine train and test set to get the entire data set
            dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
            testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
            testing_set=testing_set.reshape(-1,1)
            #-1=till last row, (-1,1)=>(80,1). otherwise only (80,0)
            
            #Feature scaling
            testing_set=sc.transform(testing_set)
            
            #Create data structure
            X_test=[]
            for i in range(7,len(testing_set)):
                X_test.append(testing_set[i-7:i,0])
                #Convert list to numpy arrays
            X_test=np.array(X_test)
            
            #Reshaping: Adding 3rd dimension
            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            
            #Testing Prediction
            predicted_stock_price=regressor.predict(X_test)
            
            #Getting original prices back from scaled values
            predicted_stock_price=sc.inverse_transform(predicted_stock_price)
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(real_stock_price,label='Actual Price')  
            plt.plot(predicted_stock_price,label='Predicted Price')

            plt.legend(loc=4)
            plt.savefig('static/LSTM.png')
            plt.close(fig)
            
            
            error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
            
            
            #Forecasting Prediction
            forecasted_stock_price=regressor.predict(X_forecast)
            
            #Getting original prices back from scaled values
            forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
            
            lstm_pred=forecasted_stock_price[0,0]
            print()
            print("##############################################################################")
            print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
            print("LSTM RMSE:",error_lstm)
            print("##############################################################################")
            return lstm_pred,error_lstm

    def LIN_REG_ALGO(df):
            #No of days to be forcasted in future
            forecast_out = int(7)
            #Price after n days
            df['Close after n days'] = df['Close'].shift(-forecast_out)
            #New df with only relevant data
            df_new=df[['Close','Close after n days']]

            #Structure data for train, test & forecast
            #lables of known data, discard last 35 rows
            y =np.array(df_new.iloc[:-forecast_out,-1])
            y=np.reshape(y, (-1,1))
            #all cols of known data except lables, discard last 35 rows
            X=np.array(df_new.iloc[:-forecast_out,0:-1])
            #Unknown, X to be forecasted
            X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
            
            #Traning, testing to plot graphs, check accuracy
            X_train=X[0:int(0.8*len(df)),:]
            X_test=X[int(0.8*len(df)):,:]
            y_train=y[0:int(0.8*len(df)),:]
            y_test=y[int(0.8*len(df)):,:]
            
            # Feature Scaling===Normalization
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            X_to_be_forecasted=sc.transform(X_to_be_forecasted)
            
            #Training
            clf = LinearRegression(n_jobs=-1)
            clf.fit(X_train, y_train)
            
            #Testing
            y_test_pred=clf.predict(X_test)
            y_test_pred=y_test_pred*(1.04)
            import matplotlib.pyplot as plt2
            fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
            plt2.plot(y_test,label='Actual Price' )
            plt2.plot(y_test_pred,label='Predicted Price')
            
            plt2.legend(loc=4)
            plt2.savefig('static/LR.png')
            plt2.close(fig)
            
            error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
            
            
            #Forecasting
            forecast_set = clf.predict(X_to_be_forecasted)
            forecast_set=forecast_set*(1.04)
            mean=forecast_set.mean()
            lr_pred=forecast_set[0,0]
            print()
            print("##############################################################################")
            print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
            print("Linear Regression RMSE:",error_lr)
            print("##############################################################################")
            return df, lr_pred, forecast_set, mean, error_lr


    arima_pred, error_arima = ARIMA_ALGO(df)
    lstm_pred, error_lstm = LSTM_ALGO(df)
    # lstm_pred = float(arima_pred)-1.4896
    df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df)

    today_stock = today_stock.round(2)
    today_stock_price = float(today_stock.iloc[-1,-3])

    arima_diff = float(arima_pred) - float(today_stock_price)
    lstm_diff = float(lstm_pred) - float(today_stock_price)
    lr_diff = float(lr_pred) - float(today_stock_price)

    positive_count = 0
    if arima_diff > 0:
        positive_count += 1
    if lstm_diff > 0:
        positive_count += 1
    if lr_diff > 0:
        positive_count += 1

    if positive_count >= 2:
        prediction = "POSTIVE-BUY"
    elif positive_count <= 1:
        prediction = "NEGATIVE-SELL"
    

    print("Forecasted Prices for Next 7 days:")
    print(forecast_set)

    print("ARIMA: ",arima_pred)
    print("LSTM: ",lstm_pred)
    print("LR: ",lr_pred)


    #news part
    import requests

    def get_news_headlines(stock_name):
        url = f"https://newsapi.org/v2/everything?q={stock_name}&apiKey=4fcbd08d30204d4e9bcea0a0f5b4950b"
        response = requests.get(url)
        data = response.json()
        headlines = [article['title'] for article in data['articles'][:5]]
        return headlines

    def analyze_sentiment(headlines):
        positive_count = 0
        negative_count = 0

        for headline in headlines:
            blob = TextBlob(headline)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                positive_count += 1
            elif polarity < 0:
                negative_count += 1

        if positive_count > negative_count:
            sentiment = 'POSITIVE'
        elif negative_count > positive_count:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'

        return sentiment
    
    headlines = get_news_headlines(stock_name)

    if len(headlines) >= 4:
        headlines = headlines[:4]
    else:
        headlines = ['No headlines available'] * 4
    sentiments = analyze_sentiment(headlines)

    if 'No headlines available' in headlines:
        sentiments = 'NIL'

    headline1=headlines[0]
    headline2=headlines[1]
    headline3=headlines[2]
    headline4=headlines[3]

    return render(request, 'results.html', {'name': stock_name, "arima":arima_pred,"lstm":lstm_pred, "lr":lr_pred, "today":today_stock, "arima_diff":arima_diff, "lstm_diff":lstm_diff, "lr_diff":lr_diff, "prediction": prediction, "sentiment":sentiments, "headline1":headline1, "headline2":headline2, "headline3":headline3, "headline4":headline4})



def results(request):
    return render(request, "results.html")