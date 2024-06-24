import pandas as pd
# dataCalendar = pd.read_csv('./data/bloc_ttf_calendar_prices.csv')

def test2():
    return 1

def getData():
    dataPrices = pd.read_csv('../../data/bloc_ttf_prices.csv')
    dataPrices['date_m'] = pd.to_datetime(dataPrices['date_m'])
    dataPrices = dataPrices.set_index('date_m')
    return dataPrices
