#my_fake_db_connect
import pymysql
class fake_db_connect():
    '''
    数据库连接工具
    '''
    def __init__(self,host:str,user:str,password:str,database:str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect(self):
        db = pymysql.connect(host=self.host,user=self.user,passwd=self.password,database=self.database)
        cursor = db.cursor()
        return db, cursor

    def get(self,primaryKey:str)->dict:
        db, cursor = self.connect()
        primaryKey = "'" + primaryKey + "'"
        sql = "SELECT * FROM userinfo where username=%s"%primaryKey
        cursor.execute(sql)
        results = cursor.fetchall()
        userdict = {
            "username": "",
            "grade": "",
            "email": "",
            "hashed_password": "",
            "money":"",
            "disabled": ""
        }
        for row in results:
            userdict["username"]=row[0]
            userdict["grade"]=row[1]
            userdict["email"]=row[2]
            userdict["hashed_password"]=row[3]
            userdict["money"]=row[4]
            userdict["disabled"]=row[5]
        db.close()
        self.userinfo = userdict
        return userdict
    
    def change(self,userdict):
        db, cursor = self.connect()
        sql = "UPDATE userinfo SET money=%f where username='%s'"%(userdict['money'],userdict['username'])
        cursor.execute(sql)
        db.commit()
        cursor.close()
        db.close()
        print("修改成功")
"""
    def insert(self,info:dict):
        primaryKey=info['username']
        self.fake_user_table[primaryKey]=info

    def commit(self):
        with open(self.filename, 'w',encoding='utf-8') as f:
            self.json.dump(self.fake_user_table, f, ensure_ascii=False,indent=2)

    def delete(self,primaryKey:str):
        del self.fake_user_table[primaryKey]
"""