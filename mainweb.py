from datetime import date
#import logging
#logging.basicConfig(level=logging.DEBUG)
#from alpha_vantage.fundamentaldata import FundamentalData
from pynse import *


nse=Nse()
import requests

import datetime

import yfinance as yf

import talib as ta

import numpy as np

import pandas as pd

from nsepy import get_history
import datetime
candle_names = ta.get_function_groups()

final = []
available_at_discount_namessr=0
final_names=[]
def intersection(arr1, arr2, arr3):
    s1 = set(arr1)

    s2 = set(arr2)

    s3 = set(arr3)

    set1 = s1.intersection(s2)  # [80, 20, 100]

    set2 = s1.intersection(s3)

    set3 = s2.intersection(s3)

    ca_and_subash = list(set1)

    ca_and_breakout = list(set2)

    subash_and_breakout = list(set3)

    subash_and_breakout = list

    print("CA AND SUBASH: ", ca_and_subash)

    print("CA and Breakout: ", ca_and_breakout)

    print("Subash and breakout: ", subash_and_breakout)

    result_set = set1.intersection(s3)

    final_list = list(result_set)

    print("SUBASH+CA+BREAKOUT", final_list)

def percentage_calc(df_x, df_y):
    return ((df_y - df_x) / df_x) * 100


#all_df=pd.read_csv("ind_nifty50list.csv")

# all_df=pd.read_csv("ind_nifty100list.csv")

#all_df = pd.read_csv('final500.csv', encoding='unicode_escape')
all_df = pd.read_csv('final500.csv')
start = datetime.date(2021, 1, 1)

end = datetime.date(2021, 6, 11)

intreval = "1d"

check = ["AVANTIFEED","BAYERCROP","BLUESTARCO"]

final_final = []
final_final_names=[]
breakout_trades = []
breakout_trades_names=[]
available_at_discount = []
available_at_discount_names=[]
quality_near_200day = []

quality_near_200day_names=[]
# supres(lo,hi)


def is_support(low, i):
    support = low[i] < low[i - 1] and low[i] < low[i +

                                                   1] and low[i + 1] < low[i + 2] and low[i - 1] < low[i - 2]

    return support


def is_resistance(high, i):
    resistance = high[i] > high[i - 1] and high[i] > high[i +

                                                          1] and high[i + 1] > high[i + 2] and high[i - 1] > high[i - 2]

    return resistance


def isfar(l, levels, s):
    return np.sum([abs(l - x) < s for x in levels]) == 0


def gave_breakout(yesterdays_close, todays_close, levels, do_print=0):
    # yesterdays_close=cl.iloc[-2]

    # todays_close=cl.iloc[-1]

    # print(levels)

    idx = (np.abs(levels - yesterdays_close))

    # print(idx)

    idx = idx.argmin()

    # print("Current price:",value)

    sup_res = []

    if (yesterdays_close > levels[idx]):

        # print("nearest support is :",array[idx])

        sup_res.append(levels[idx])

        if (idx < len(levels) - 1):

            sup_res.append(levels[idx + 1])

            # print("nearest resistance is:",array[idx+1])

        else:

            sup_res.append(0)  # its at its aTH



    else:

        sup_res.append(levels[idx - 1])

        sup_res.append(levels[idx])

    if (todays_close >= sup_res[1] and do_print == 1):

        if (sup_res[1] == 0):
            print("ALL TIME high:", todays_close)

        print("Broke resistance of:", sup_res[1])

        for i in range(idx - 2, len(levels)):
            print(levels[i], end=" ")

        print(sup_res)

    if (todays_close >= sup_res[1]):
        # print("Broke resistance of ",sup_res[1])

        return 1

    return 0


def find_levels(low, high):
    levels = []

    levels = np.array(levels)

    s = np.mean(high - low)

    for i in range(2, len(low) - 2):

        if (is_support(low, i)):

            l = low[i]

            if (isfar(l, levels, s)):
                levels = np.append(levels, l)

                # levels.append(l)

        elif (is_resistance(high, i)):

            l = high[i]

            if (isfar(l, levels, s)):
                levels = np.append(levels, l)

    levels.sort()

    return levels

    # print(levels)


final_and_breakout = []
final_and_breakout_names=[]
    current_date = datetime.datetime.now()
def find_trades():
    finalsr = 0
    final_finalsr = 0
    final_and_breakoutsr = 0
    quality_near_200daysr = 0
    breakout_tradessr = 0
    available_at_discount_namessr=0
    for names in all_df['Symbol']:

    #for names in check:

        print(names)

        symbol = names

        levels = []

        # df=yf.Ticker(f"{symbol}.NS").history(start=start,end=end,interval=intreval)
        df=nse.get_hist(names, from_date=datetime.date(2021,10,1),to_date=datetime.date(current_date.year, current_date.month, current_date.day))
        
        #df = get_history(symbol=names, start=date(2020, 10, 1), end=date(2021, 12, 7))
        #df = get_history(symbol=names, start=date(2020, 10, 1), end=date(current_date.year, current_date.month, current_date.day))
        #print(df.head())
        df['high(7)'] = df['close'].rolling(7).max().shift(-7)

        # df['high(7)'].fillna(method='ffill',inplace=True)

        df['volume(4)'] = df['volume'].rolling(

            window=4).max().shift(1).fillna(0)

        df['MOVING_volume'] = ta.SMA(df['volume'], timeperiod=5)

        op = df['open']

        hi = df['high']

        lo = df['low']

        cl = df['close']

        df['Morning_Star'] = ta.CDLMORNINGSTAR(op, hi, lo, cl)

        df['Engulfing'] = ta.CDLENGULFING(op, hi, lo, cl)

        df['Hammer'] = ta.CDLHAMMER(op, hi, lo, cl)

        # num=ta.CDLMORNINGSTAR(op,hi,lo,cl)

        all_time_high = cl.max()

        min_limit = 0.9 * all_time_high

        df['EMA5'] = ta.EMA(df['close'], timeperiod=5)

        df['EMA13'] = ta.EMA(df['close'], timeperiod=13)

        df['EMA26'] = ta.EMA(df['close'], timeperiod=26)

        df['EMA12'] = ta.EMA(df['close'], timeperiod=12)

        df['EMA200'] = ta.EMA(df['close'], timeperiod=200)

        df['MACD'] = df['EMA12'] - df['EMA26']

        df['SIGNAL'] = ta.EMA(df['MACD'], timeperiod=9)

        df["RSI"] = ta.RSI(df['close'], timeperiod=14)

        df['RSI_SLOPE'] = df['RSI'] - df['RSI'].shift(1)

        df['RSI_DIFF'] = df['RSI_SLOPE'] - df['RSI_SLOPE'].shift(1)

        df['HISTO'] = df['MACD'] - df['SIGNAL']

        df["yesterdays_high"] = df['high'].shift(1)

        df["yesterdays_close"] = df['close'].shift(1)

        df['yesterdays_ema5'] = df['EMA5'].shift(1)

        # df["prev_under_5?"]=np.where((df['EMA5']>cl),1,0)

        df['HISTO_DIFF'] = df['HISTO'] - df['HISTO'].shift(1)

        df['ROC'] = ta.ROC(df['close'], 9)

        df['EMA buy?'] = np.where(

            (df['EMA5'] > df['EMA13']) & (df['EMA5'] > df['EMA26']), 1, 0)

        df['MACD buy?'] = np.where((df['HISTO_DIFF'] > 0), 1, 0)

        # df['MACD buy?']=np.where( (df['HISTO_DIFF']>0)&(df['HISTO']<=0), 1, 0)  #dont use when market is overvalued

        df["RSI buy?"] = np.where((df['RSI'] >= 40) & (

                df['RSI'] <= 70) & (df['RSI_DIFF'] > 0), 1, 0)

        df['volume buy?'] = np.where(

            (df['volume'] >= df['MOVING_volume']), 1, 0)

        df['ROC buy?'] = np.where((df['ROC'] > 0), 1, 0)
        df['day_percentage']=percentage_calc(df['close'],df['close'].shift(1))
        df['percentage'] = percentage_calc(df['close'], df['high(7)'])

        df["Final buy?"] = np.where((df['EMA buy?'] == 1) & (df['MACD buy?'] == 1) & (

                df['RSI buy?'] == 1) & (df['volume buy?'] == 1) & (df['ROC buy?'] == 1), 1, 0)

        df['yesterdays_high_under_ema5?'] = np.where(

            (df['yesterdays_high'] < df['yesterdays_ema5']), 1, 0)

        df["power_buy?"] = np.where(

            (df['yesterdays_high_under_ema5?'] == 1) & (cl > df['yesterdays_high']), 1, 0)

        df['Bear_value_buy'] = np.where(df['close'] <= min_limit, 1, 0)

        df['near200ema?'] = np.where(

            abs((df['close'] - df['EMA200']) / df['EMA200']) <= 0.02, 1, 0)

        levels = find_levels(lo, hi)

        # levels = np.asarray(levels)

        # df['gave_breakout']=df.apply(gave_breakout,args=(df['yesterdays_close'],df['close']),axis=1)

        df['gave_breakout'] = df.apply(lambda x: gave_breakout(

            x['yesterdays_close'], x['close'], levels), axis=1)

        df['breakout_buy?'] = np.where(

            (df['volume buy?'] == 1) & (df['gave_breakout'] == 1), 1, 0)

        df['final_and_breakout?'] = np.where(

            (df['breakout_buy?'] == 1) & (df['Final buy?'] == 1), 1, 0)

        df['worked?'] = np.where(

            (percentage_calc(df['close'], df['high(7)']) > 3), 1, 0)

        # given_breakout = gave_breakout(cl, levels)

        # if(given_breakout):

        #       if((df['volume buy?'].iloc[-1]==1)and  (    ((df['close'].iloc[-1]-df['open'].iloc[-1])   /df['open'].iloc[-1]  )>=0.03     )):

        #              print(names," has also good volumes")

        #             breakout_trades.append(names)
        def put_into_presentation(array_name,names,sr):
            that_row = all_df.loc[all_df['Symbol'] == names]
            temp = that_row.to_numpy()
            #sr+=1
            array_name.append([sr,temp[0][0], names, temp[0][1], df["close"].iloc[-1],
                          df["volume"].iloc[-1]])
        if (df['near200ema?'].iloc[-1] == 1):
            print(names, "is near 200 day moving average")
            quality_near_200daysr=quality_near_200daysr+1
            put_into_presentation(quality_near_200day,names,quality_near_200daysr)
            #quality_near_200day.append(names)
            quality_near_200day_names.append(names)

        if (df['final_and_breakout?'].iloc[-1] == 1):
            print(names, " can be bought acc to breakout and CA rachana")
            final_and_breakoutsr=final_and_breakoutsr+1
            put_into_presentation(final_and_breakout, names,final_and_breakoutsr)
            final_and_breakout_names.append(names)
            #final_and_breakout.append(names)

        if (df['breakout_buy?'].iloc[-1] == 1):
            gave_breakout(df['yesterdays_close'].iloc[-1],

                          df['close'].iloc[-1], levels, 1)
            breakout_trades_names.append(names)
            breakout_tradessr=breakout_tradessr+1
            #breakout_trades.append(names)
            put_into_presentation(breakout_trades, names,breakout_tradessr)

        if (df['Bear_value_buy'].iloc[-1] == 1):
            # print(names,"is trading at a discount of : ",(((all_time_high- df['close'].iloc[-1]) /all_time_high   )*100),"from its all time high" )
            available_at_discount_namessr=available_at_discount_namessr+1
            put_into_presentation(available_at_discount,names,available_at_discount_namessr)
            #available_at_discount.append(names)
            available_at_discount_names.append(names)
        # df.dropna(inplace=True)

        if (df['Final buy?'].iloc[-1] == 1):
            print(names, "can buy acc to ca rachana")
            finalsr=finalsr+1
            put_into_presentation(final,names,finalsr)
            #final.append([temp[0][0],names,temp[0][1],df["close"].iloc[-1],df['day_percentage'].iloc[-1],df["volume"].iloc[-1]])
            final_names.append(names)
        if (df["power_buy?"].iloc[-1] == 1):
            print(names, "can buy acc to SUBASH")
            final_finalsr=final_finalsr+1
            put_into_presentation(final_final,names,final_finalsr)
            #final_final.append(names)
            final_final_names.append(names)


    print(final, final_final, final_and_breakout)
    intersection(final_names, final_final_names, breakout_trades_names)
    return [final, final_final, final_and_breakout,quality_near_200day,breakout_trades]


print("Near 200 day moving average:", quality_near_200day)




find_trades()




def pitroski_score():
    key = '6858QKN1AXBAT9GM'

    url = 'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=BSE:TITAN&apikey=6858QKN1AXBAT9GM'

    r = requests.get(url)

    data = r.json()

    print(data)


pitroski_score()
