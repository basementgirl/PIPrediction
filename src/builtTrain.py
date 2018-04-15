import pandas as pd
from builtSession import *



def buildTrain(sessions) :
    # sessions = [[123,111,234],[123,323,4234],[2,23,234],[2,31,323]]
    # sessions = pd.DataFrame(sessions,columns=('session_id','sku_id','type'))

    groups = sessions.groupby('session_id')
    for group in groups :
        session_id,session = group
        print( session.groupby('sku_id').count() )

        break



if __name__ == '__main__' :
    data = pre_deal('data/temp/action.csv')
    sessions = builtSession(data)
    buildTrain(sessions)
