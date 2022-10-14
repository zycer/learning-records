import MySQLdb


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
