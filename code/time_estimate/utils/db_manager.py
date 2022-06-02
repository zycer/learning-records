import logging
import MySQLdb
from MySQLdb import cursors, _mysql
from MySQLdb._exceptions import Error as MySQLdbError
#
#
# class DBManager:
#     """
#     1
#     sqluser = DBQuery().ment("SELECT * FROM `user` ")  # 直接使用
#
#     2
#     sqldb = DBQuery()  # sql 初始化
#     fimes = sqldb.ment("SELECT * FROM tables WHERE username=%s AND id=%s", (usernmae, 1), close=False)  # 不关当前链接
#
#     3
#     connectdata = {'host': '127.0.0.1'}
#     sqldb = DBQuery(**connectdata) # sql 初始化
#     fimes = sqldb.ment("SELECT * FROM tables WHERE username=%s AND id=%s", (usernmae, 1))  # 自动关闭当前链接
#
#     """
#
#     def __init__(self, **kwargs):
#         self.mysql_config = {
#             'user': 'root',
#             'passwd': '123456',
#             'host': '127.0.0.1',
#             'db': 'ev_estimate',
#             'port': 3306,
#             'connect_timeout': 10,
#             'cursorclass': cursors.DictCursor,
#         }
#         self.kwargs = kwargs
#         self.replaceconfig()
#         self.config = self.mysql_config.copy()
#         self.dictionary = True  # 返回类型 dict, 默认类型[()]
#         self.conn = None
#         self.cur = None
#
#     def replaceconfig(self):
#         """ 替换参数 """
#         for k, v in self.kwargs.items():
#             self.mysql_config[k] = v
#
#     def connect(self):
#         """ 数据库连接函数,建立连接和游标 """
#         if self.conn and self.cur:
#             return True
#
#         try:
#             self.conn = MySQLdb.connect(**self.config)
#         except MySQLdbError:
#             logging.error("数据库连接失败", exc_info=True)
#             return False
#         self.cur = self.conn.cursor()
#         return True
#
#     def ment(self, statement, parameter=None, close=True):
#         """ 数据执行语句函数 ,statement:语句,parameter:参数,必须是 元组 或者 列表,返回执行结果"""
#         # print(statement,parameter)
#         if not self.connect():
#             return False
#         try:
#             self.cur.execute(statement)
#
#             if 'SELECT' == statement.strip().upper()[0:6] or 'SHOW' == statement.strip().upper()[0:4]:
#                 result = self.cur.fetchall()
#             elif 'CREATE TABLE' == statement.strip().upper()[0:12]:
#                 result = True
#             else:
#                 result = self.conn.commit()
#                 if statement.upper()[0:6] in ['INSERT', 'UPDATE']:
#                     result = self.rowcount()
#         except MySQLdbError as e:
#             logging.error('SQL_ERROR: %s, statement:%s' % (e, statement))
#             self.conn.rollback()
#             result = False
#         finally:
#             self.mclose(close)
#         return result
#
#     def lastd(self):
#         """ 返回 INSERT 语句的id, 应该在插入后立即使用 """
#         try:
#             return self.cur.lastrowid
#         except:
#             pass
#
#     def rowcount(self):
#         """ 返回 INSERT or UPDATE 语句的更新数目 """
#         try:
#             return self.cur.rowcount
#         except:
#             pass
#
#     def mclose(self, close=True):
#         if close:
#             self.dbclose()
#
#     def dbclose(self):
#         try:
#             if self.cur:
#                 self.cur.close()
#             if self.conn:
#                 self.conn.close()
#             return True
#         except:
#             pass


class DBManager:
    def __init__(self):
        self.conn = MySQLdb.connect(
            host='localhost',
            port=3306,
            user='root',
            passwd='123456',
            db='ev_estimate'
        )
        self.cur = self.conn.cursor()

    def exec_sql(self, sql):
        self.cur.execute(sql)
        if "SELECT" not in sql:
            self.conn.commit()
        return self.cur.fetchall()

    def close(self):
        self.cur.close()
        self.conn.close()

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32,
                                      np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):  # add this line
                    return obj.tolist()  # add this line
                return json.JSONEncoder.default(self, obj)



