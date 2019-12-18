from sqlalchemy import Column, create_engine, INTEGER, CHAR, NVARCHAR, DATETIME, Sequence
from sqlalchemy.ext.declarative import declarative_base

# 创建对象的基类
Base = declarative_base()


class Table1(Base):
    # 表的名字:
    __tablename__ = 'Table_el1'

    # 表的结构:
    LOTNO = Column(NVARCHAR(50), primary_key=True)
    USERNO = Column(NVARCHAR(50))
    EQUIPMENTNO = Column(NVARCHAR(50))
    QUALITY = Column(NVARCHAR(4))
    FILEPATH = Column(NVARCHAR(255))
    FILEPATH2 = Column(NVARCHAR(255))
    CREATEDATE = Column(DATETIME)
    TESTDATE = Column(DATETIME, primary_key=True)
    REMARK = Column(NVARCHAR(255))
    UPDATEFLAG = Column(INTEGER)


class Table2(Base):
    # 表的名字:
    __tablename__ = 'Table_el2'

    # 表的结构:
    LOTNO = Column(NVARCHAR(50), primary_key=True)
    CREATEDATE = Column(DATETIME, primary_key=True)
    QUALITY = Column(NVARCHAR(4))
    DEFECT1 = Column(NVARCHAR(255))
    DEFECT2 = Column(NVARCHAR(255))
    DEFECT3 = Column(NVARCHAR(255))
    DEFECT4 = Column(NVARCHAR(255))
    DEFECT5 = Column(NVARCHAR(255))
    DEFECT6 = Column(NVARCHAR(255))
    DEFECT7 = Column(NVARCHAR(255))
    DEFECT8 = Column(NVARCHAR(255))
    DEFECT9 = Column(NVARCHAR(255))
    DEFECT10 = Column(NVARCHAR(255))
    DEFECT11 = Column(NVARCHAR(255))
    DEFECT12 = Column(NVARCHAR(255))
    DEFECT13 = Column(NVARCHAR(255))
    DEFECT14 = Column(NVARCHAR(255))
    DEFECT15 = Column(NVARCHAR(255))
    DEFECT16 = Column(NVARCHAR(255))
    DEFECT17 = Column(NVARCHAR(255))
    DEFECT18 = Column(NVARCHAR(255))
    DEFECT19 = Column(NVARCHAR(255))
    DEFECT20 = Column(NVARCHAR(255))
    DEFECT21 = Column(NVARCHAR(255))
    DEFECT22 = Column(NVARCHAR(255))
    DEFECT23 = Column(NVARCHAR(255))
    DEFECT24 = Column(NVARCHAR(255))


class TableLog(Base):
    # 表的名字:
    __tablename__ = 'TableLog'

    Id = Column(INTEGER, Sequence('log_id'), nullable=False, primary_key=True)
    LotSN = Column(NVARCHAR(50))
    Flage = Column(CHAR(1))
    Txml = Column(NVARCHAR(4000))


# 初始化数据库连接
engine = create_engine('mssql+pymssql://sa:123456@localhost:1434/longi_test', encoding="utf-8")
# 查看sqlserver端口
# exec sys.sp_readerrorlog 0, 1, 'listening'

# 创建表
Base.metadata.create_all(engine)

# 关闭数据库连接
engine.dispose()