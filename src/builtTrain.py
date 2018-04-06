import pandas as pd



def builtSession( ) :
    sessions = [[123,111,234],[123,323,4234],[2,23,234],[2,31,323]]
    sessions = pd.DataFrame(sessions,columns=('session_id','sku_id','type'))

    sessionGroup = sessions.groupby(['session_id','type'])
    for session in sessionGroup :
        print(session)
        break



if __name__ == '__main__' :
    builtSession()
    print('aaa')