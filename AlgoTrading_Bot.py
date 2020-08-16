from kiteconnect import KiteConnect
import os
import datetime as dt
import pandas as pd
import numpy as np
import time
from stocktrends import Renko
import sys
import statsmodels.api as sm

cwd = os.chdir("D:\\zerodha")

#generate trading session
access_token = open("access_token.txt",'r').read()
key_secret = open("api_key.txt",'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)


#get dump of all NSE instruments
instrument_dump = kite.instruments("NSE")
instrument_df = pd.DataFrame(instrument_dump)


def instrumentLookup(instrument_df,symbol):
    """Looks up instrument token for a given script from instrument dump"""
    try:
        return instrument_df[instrument_df.tradingsymbol==symbol].instrument_token.values[0]
    except:
        return -1

def tickerLookup(token):
    global instrument_df
    return instrument_df[instrument_df.instrument_token==token].tradingsymbol.values[0]


def fetchOHLC(ticker,interval,duration):
    """extracts historical data and outputs in the form of dataframe"""
    instrument = instrumentLookup(instrument_df,ticker)
    data = pd.DataFrame(kite.historical_data(instrument,dt.date.today()-dt.timedelta(duration), dt.date.today(),interval))
    data.set_index("date",inplace=True)
    return data


def atr(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['high']-df['low'])
    df['H-PC']=abs(df['high']-df['close'].shift(1))
    df['L-PC']=abs(df['low']-df['close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].ewm(com=n,min_periods=n).mean()
    return df['ATR']


def supertrend(DF,n,m):
    """function to calculate Supertrend given historical candle data
        n = n day ATR - usually 7 day ATR is used
        m = multiplier - usually 2 or 3 is used"""
    df = DF.copy()
    df['ATR'] = atr(df,n)
    df["B-U"]=((df['high']+df['low'])/2) + m*df['ATR'] 
    df["B-L"]=((df['high']+df['low'])/2) - m*df['ATR']
    df["U-B"]=df["B-U"]
    df["L-B"]=df["B-L"]
    ind = df.index
    for i in range(n,len(df)):
        if df['close'][i-1]<=df['U-B'][i-1]:
            df.loc[ind[i],'U-B']=min(df['B-U'][i],df['U-B'][i-1])
        else:
            df.loc[ind[i],'U-B']=df['B-U'][i]    
    for i in range(n,len(df)):
        if df['close'][i-1]>=df['L-B'][i-1]:
            df.loc[ind[i],'L-B']=max(df['B-L'][i],df['L-B'][i-1])
        else:
            df.loc[ind[i],'L-B']=df['B-L'][i]  
    df['Strend']=np.nan
    for test in range(n,len(df)):
        if df['close'][test-1]<=df['U-B'][test-1] and df['close'][test]>df['U-B'][test]:
            df.loc[ind[test],'Strend']=df['L-B'][test]
            break
        if df['close'][test-1]>=df['L-B'][test-1] and df['close'][test]<df['L-B'][test]:
            df.loc[ind[test],'Strend']=df['U-B'][test]
            break
    for i in range(test+1,len(df)):
        if df['Strend'][i-1]==df['U-B'][i-1] and df['close'][i]<=df['U-B'][i]:
            df.loc[ind[i],'Strend']=df['U-B'][i]
        elif  df['Strend'][i-1]==df['U-B'][i-1] and df['close'][i]>=df['U-B'][i]:
            df.loc[ind[i],'Strend']=df['L-B'][i]
        elif df['Strend'][i-1]==df['L-B'][i-1] and df['close'][i]>=df['L-B'][i]:
            df.loc[ind[i],'Strend']=df['L-B'][i]
        elif df['Strend'][i-1]==df['L-B'][i-1] and df['close'][i]<=df['L-B'][i]:
            df.loc[ind[i],'Strend']=df['U-B'][i]
    return df['Strend']




def st_dir_refresh(renko,ticker):
    """function to check for supertrend reversal"""
    global st_dir
    if renko[ticker]["st1"].tolist()[-1] > renko[ticker]["close"].tolist()[-1] and renko[ticker]["st1"].tolist()[-2] < renko[ticker]["close"].tolist()[-2] :
        st_dir[ticker][0] = "red"
    if renko[ticker]["st2"].tolist()[-1] > renko[ticker]["close"].tolist()[-1] and renko[ticker]["st2"].tolist()[-2] < renko[ticker]["close"].tolist()[-2] :
        st_dir[ticker][1] = "red"
    if renko[ticker]["st3"].tolist()[-1] > renko[ticker]["close"].tolist()[-1] and renko[ticker]["st3"].tolist()[-2] < renko[ticker]["close"].tolist()[-2] :
        st_dir[ticker][2] = "red"
    if renko[ticker]["st1"].tolist()[-1] < renko[ticker]["close"].tolist()[-1] and renko[ticker]["st1"].tolist()[-2] > renko[ticker]["close"].tolist()[-2] :
        st_dir[ticker][0] = "green"
    if renko[ticker]["st2"].tolist()[-1] < renko[ticker]["close"].tolist()[-1] and renko[ticker]["st2"].tolist()[-2] > renko[ticker]["close"].tolist()[-2] :
        st_dir[ticker][1] = "green"
    if renko[ticker]["st3"].tolist()[-1] < renko[ticker]["close"].tolist()[-1] and renko[ticker]["st3"].tolist()[-2] > renko[ticker]["close"].tolist()[-2] :
        st_dir[ticker][2] = "green"


def st_dir_refresh_5(renko_5,ticker):
    """function to check for supertrend reversal"""
    global st_dir_5
    if renko_5[ticker]["st1"].tolist()[-1] > renko_5[ticker]["close"].tolist()[-1] :
        st_dir_5[ticker][0] = "red"
    if renko_5[ticker]["st2"].tolist()[-1] > renko_5[ticker]["close"].tolist()[-1] :
        st_dir_5[ticker][1] = "red"
    if renko_5[ticker]["st3"].tolist()[-1] > renko_5[ticker]["close"].tolist()[-1] :
        st_dir_5[ticker][2] = "red"
    if renko_5[ticker]["st1"].tolist()[-1] < renko_5[ticker]["close"].tolist()[-1] :
        st_dir_5[ticker][0] = "green"
    if renko_5[ticker]["st2"].tolist()[-1] < renko_5[ticker]["close"].tolist()[-1] :
        st_dir_5[ticker][1] = "green"
    if renko_5[ticker]["st3"].tolist()[-1] < renko_5[ticker]["close"].tolist()[-1] :
        st_dir_5[ticker][2] = "green"

def st_dir_refresh_15(renko_15,ticker):
    """function to check for supertrend reversal"""
    global st_dir_15

    if renko_15[ticker]["st1"].tolist()[-1] > renko_15[ticker]["close"].tolist()[-1] :
        st_dir_15[ticker][0] = "red"
    if renko_15[ticker]["st2"].tolist()[-1] > renko_15[ticker]["close"].tolist()[-1] :
        st_dir_15[ticker][1] = "red"
    if renko_15[ticker]["st3"].tolist()[-1] > renko_15[ticker]["close"].tolist()[-1] :
        st_dir_15[ticker][2] = "red"
    if renko_15[ticker]["st1"].tolist()[-1] < renko_15[ticker]["close"].tolist()[-1] :
        st_dir_15[ticker][0] = "green"
    if renko_15[ticker]["st2"].tolist()[-1] < renko_15[ticker]["close"].tolist()[-1] :
        st_dir_15[ticker][1] = "green"
    if renko_15[ticker]["st3"].tolist()[-1] < renko_15[ticker]["close"].tolist()[-1] :
        st_dir_15[ticker][2] = "green"


def placeSLOrder(symbol,buy_sell,quantity,sl_price):    
    # Place an intraday stop loss order on NSE
    if buy_sell == "buy":
        t_type=kite.TRANSACTION_TYPE_BUY
        t_type_sl=kite.TRANSACTION_TYPE_SELL
    elif buy_sell == "sell":
        t_type=kite.TRANSACTION_TYPE_SELL
        t_type_sl=kite.TRANSACTION_TYPE_BUY
    kite.place_order(tradingsymbol=symbol,
                    exchange=kite.EXCHANGE_NSE,
                    transaction_type=t_type,
                    quantity=quantity,
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_MIS,
                    variety=kite.VARIETY_REGULAR)
    kite.place_order(tradingsymbol=symbol,
                    exchange=kite.EXCHANGE_NSE,
                    transaction_type=t_type_sl,
                    quantity=quantity,
                    order_type=kite.ORDER_TYPE_SL,
                    price=sl_price,
                    trigger_price = sl_price,
                    product=kite.PRODUCT_MIS,
                    variety=kite.VARIETY_REGULAR)


def ModifyOrder(order_id,price):    
    # Modify order given order id
    kite.modify_order(order_id=order_id,
                    price=price,
                    trigger_price=price,
                    order_type=kite.ORDER_TYPE_SL,
                    variety=kite.VARIETY_REGULAR) 

def placeMarketOrder(symbol,buy_sell,quantity):    
    # Place an intraday market order on NSE
    if buy_sell == "buy":
        t_type=kite.TRANSACTION_TYPE_BUY
    elif buy_sell == "sell":
        t_type=kite.TRANSACTION_TYPE_SELL
    kite.place_order(tradingsymbol=symbol,
                    exchange=kite.EXCHANGE_NSE,
                    transaction_type=t_type,
                    quantity=quantity,
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_MIS,
                    variety=kite.VARIETY_REGULAR)

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df2 = Renko(df)
    #df2.brick_size = min(10,round((df["close"].tolist()[-1]*.0025),2))
    df2.brick_size = min(10,round((0.05 *round(float(df["close"].tolist()[-1]*0.0025)/0.05)),2))
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    return renko_df     

def main(capital):
    a,b = 0,0
    while a < 10:
        try:
            pos_df = pd.DataFrame(kite.positions()["day"])
            break
        except:
            print("can't extract position data..retrying")
            a+=1
    while b < 10:
        try:
            ord_df = pd.DataFrame(kite.orders())
            break
        except:
            print("can't extract order data..retrying")
            b+=1
    
    for ticker in tickers:
        print("starting passthrough for.....",ticker)
        try:
            ohlc[ticker]  = fetchOHLC(ticker,"minute",5)
            renko[ticker]  = renko_DF(ohlc[ticker])
            renko[ticker]["st1"] = supertrend(renko[ticker],7,3)
            renko[ticker]["st2"] = supertrend(renko[ticker],10,3)
            renko[ticker]["st3"] = supertrend(renko[ticker],11,2)

            ohlc_5[ticker] = fetchOHLC(ticker,"5minute",10)
            renko_5[ticker]  = renko_DF(ohlc_5[ticker])
            renko_5[ticker]["st1"] = supertrend(renko_5[ticker],7,3)
            renko_5[ticker]["st2"] = supertrend(renko_5[ticker],10,3)
            renko_5[ticker]["st3"] = supertrend(renko_5[ticker],11,2)

            ohlc_15[ticker] = fetchOHLC(ticker,"15minute",10)
            renko_15[ticker]  = renko_DF(ohlc_15[ticker])
            renko_15[ticker]["st1"] = supertrend(renko_15[ticker],7,3)
            renko_15[ticker]["st2"] = supertrend(renko_15[ticker],10,3)
            renko_15[ticker]["st3"] = supertrend(renko_15[ticker],11,2)


            st_dir_refresh(renko,ticker)
            st_dir_refresh_5(renko_5,ticker)
            st_dir_refresh_15(renko_15,ticker)

            #quantity = int(capital/ohlc["close"][-1])
            quantity=1
            if len(pos_df.columns)==0:
                if st_dir[ticker] == ["green","green","green"] and st_dir_5[ticker] == ["green","green","green"] and st_dir_15[ticker] == ["green","green","green"] and renko[ticker]["bar_num"].tolist()[-1] >=1 :
                    placeSLOrder(ticker,"buy",quantity, round((0.05 *round(float(renko[ticker]["low"].tolist()[-1])/0.05)),2))
                if st_dir[ticker] == ["red","red","red"] and st_dir_5[ticker] == ["red","red","red"] and st_dir_15[ticker] == ["red","red","red"] and renko[ticker]["bar_num"].tolist()[-1] <=-1 :
                    placeSLOrder(ticker,"sell",quantity, round((0.05 *round(float(renko[ticker]["high"].tolist()[-1])/0.05)),2))
            if len(pos_df.columns)!=0 and ticker not in pos_df["tradingsymbol"].tolist():
                if st_dir[ticker] == ["green","green","green"] and st_dir_5[ticker] == ["green","green","green"] and st_dir_15[ticker] == ["green","green","green"] and renko[ticker]["bar_num"].tolist()[-1] >=1 :
                    placeSLOrder(ticker,"buy",quantity, round((0.05 *round(float(renko[ticker]["low"].tolist()[-1])/0.05)),2) )
                if st_dir[ticker] == ["red","red","red"] and st_dir_5[ticker] == ["red","red","red"] and st_dir_15[ticker] == ["red","red","red"] and renko[ticker]["bar_num"].tolist()[-1] <=-1 :
                    placeSLOrder(ticker,"sell",quantity, round((0.05 *round(float(renko[ticker]["high"].tolist()[-1])/0.05)),2))
            if len(pos_df.columns)!=0 and ticker in pos_df["tradingsymbol"].tolist():
                if pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0] == 0:
                    if st_dir[ticker] == ["green","green","green"] and st_dir_5[ticker] == ["green","green","green"] and st_dir_15[ticker] == ["green","green","green"] and renko[ticker]["bar_num"].tolist()[-1] >=1 :
                        placeSLOrder(ticker,"buy",quantity, round((0.05 *round(float(renko[ticker]["low"].tolist()[-1])/0.05)),2) )
                    if st_dir[ticker] == ["red","red","red"] and st_dir_5[ticker] == ["red","red","red"] and st_dir_15[ticker] == ["red","red","red"] and renko[ticker]["bar_num"].tolist()[-1] <=-1 :
                        placeSLOrder(ticker,"sell",quantity, round((0.05 *round(float(renko[ticker]["high"].tolist()[-1])/0.05)),2))
                if pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0] > 0:
                    if (((ord_df[ord_df["tradingsymbol"]==ticker]['status'].isin(["TRIGGER PENDING","OPEN"])).values[-1]) == False)==False:
                        
                        order_id = ord_df.loc[(ord_df['tradingsymbol'] == ticker) & (ord_df['status'].isin(["TRIGGER PENDING","OPEN"]))]["order_id"].values[0]
                        ModifyOrder(order_id,round((0.05 *round(float(renko[ticker]["low"].tolist()[-1])/0.05)),2))
                if pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0] < 0:
                    if (((ord_df[ord_df["tradingsymbol"]==ticker]['status'].isin(["TRIGGER PENDING","OPEN"])).values[-1]) == False)==False:
                        order_id = ord_df.loc[(ord_df['tradingsymbol'] == ticker) & (ord_df['status'].isin(["TRIGGER PENDING","OPEN"]))]["order_id"].values[0]
                        ModifyOrder(order_id,round((0.05 *round(float(renko[ticker]["high"].tolist()[-1])/0.05)),2))
                
                if pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0] > 0 and renko[ticker]["bar_num"].tolist()[-1] <=-1 :
                    if (((ord_df[ord_df["tradingsymbol"]==ticker]['status'].isin(["TRIGGER PENDING","OPEN"])).values[-1]) == False):
                    #order_id = ord_df.loc[(ord_df['tradingsymbol'] == ticker) & (ord_df['status'].isina(["TRIGGER PENDING","OPEN"]))]["order_id"].values[0]
                        placeMarketOrder(ticker,"sell",pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0])
                if pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0] < 0 and renko[ticker]["bar_num"].tolist()[-1] >=1 :
                    if (((ord_df[ord_df["tradingsymbol"]==ticker]['status'].isin(["TRIGGER PENDING","OPEN"])).values[-1]) == False):
                    #order_id = ord_df.loc[(ord_df['tradingsymbol'] == ticker) & (ord_df['status'].isina(["TRIGGER PENDING","OPEN"]))]["order_id"].values[0]
                        placeMarketOrder(ticker,"buy",abs(pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0]))    
                
        except:
            print("API error for ticker :",ticker)
            try:
                if pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0] > 0:
                    placeMarketOrder(ticker,"sell",pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0])
                if pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0] < 0:
                    placeMarketOrder(ticker,"buy",abs(pos_df[pos_df["tradingsymbol"]==ticker]["quantity"].values[0]))
            except:
                print("API error for ticker :",ticker)
                
                
            
            
#############################################################################################################
###################################################################################################"######"####
tickers = ["ZEEL","JSWSTEEL","AXISBANK","BPCL","INDUSINDBK","ITC","HDFCBANK","TATASTEEL",
			"HDFC","ADANIPORTS","TITAN","UPL","CIPLA","DRREDDY","GRASIM","BRITANNIA",
			"BHARTIARTL","INFRATEL","HCLTECH"]

#tickers to track - recommended to use max movers from previous day
capital = 3000 #position size
st_dir = {}
st_dir_5 = {}
st_dir_15 = {}

ohlc={}
ohlc_5={}
ohlc_15={}

renko={}
renko_5={}
renko_15={}

ord_df=[]
pos_df =[] #directory to store super trend status for each ticker
for ticker in tickers:
    st_dir[ticker] = ["None","None","None"]    
    st_dir_5[ticker] = ["None","None","None"]
    st_dir_15[ticker] = ["None","None","None"]



while True:
    now = dt.datetime.now()
    if (now.hour >= 5 and now.minute >= 45):
        starttime=time.time()
        timeout = time.time() + 60*60*8  # 60 seconds times 360 meaning 6 hrs
        while time.time() <= timeout:
            try:
                main(capital)
                time.sleep(20 - ((time.time() - starttime) % 20.0))
            except KeyboardInterrupt:
                print('\n\nKeyboard exception received. Exiting.')
                exit()  
        
    if (now.hour >= 22 and now.minute >= 30):
        sys.exit()