import gradio as gr
from main import tokencreator,db

class Datarw():
    def __init__(self, tokencreator, database) -> None:
        self.tokencreator = tokencreator
        self.database = database

    def readinfo(self,once_token):
        # 尝试解码token获取username
        username = self.tokencreator.decode(once_token)
        if username is None: raise gr.Error("You are not login, please login!")
	    # 尝试从数据库取用户信息
        try: userinfo = self.database.get(username)
        except: raise gr.Error("You are not login, please login!")
        if userinfo is None: raise gr.Error("You are not login, please login!")
        print(userinfo['money'])#显示用户余额
        self.userinfo = userinfo
    
    def writeinfo(self,once_token,spend):
        self.readinfo(once_token=once_token)
        self.userinfo['money'] = self.userinfo['money'] - spend
        db.change(self.userinfo) # 从my_fake_db_connect导入函数
    
    def writeinfo_with_username(self, username, spend):
        # 尝试从数据库取用户信息
        try: userinfo = self.database.get(username)
        except: raise gr.Error("You are not login, please login!")
        if userinfo is None: raise gr.Error("You are not login, please login!")
        self.userinfo = userinfo
        self.userinfo['money'] = self.userinfo['money'] - spend
        db.change(self.userinfo) # 从my_fake_db_connect导入函数

datarw = Datarw(tokencreator=tokencreator, database=db)