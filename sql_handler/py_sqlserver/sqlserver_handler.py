import os
import datetime as dt

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sql_handler.py_sqlserver.sqlserver_engine import Table1, TableLog

engine = create_engine('mssql+pymssql://sa:123456@127.0.0.1:1434/longi_test')

# 创建DBSession类型:
DBSession = sessionmaker(bind=engine)
Session = DBSession()  # 实例化与数据库的会话

modify_time = os.path.getmtime(r"G:\longi\产线图片\20190710\CFlight\NG\LRP504033190700229723.jpg")
t1 = Table1(LOTNO='LRP504033190700229724', QUALITY='NG',
            FILEPATH=r'D:\ccc\M1EL-202\20180720\DayFlight\OK\LRP504033190700229723.jpg',
            CREATEDATE=dt.datetime.now(), TESTDATE=dt.datetime.fromtimestamp(modify_time),
            UPDATEFLAG=0)

Session.add(t1)

Session.add_all([
    TableLog(LotSN='asdfadf'),
    TableLog(LotSN='qwerqwe'),

])

Session.commit()  # 提交，不然不能创建数据
Session.close()

engine.dispose()
