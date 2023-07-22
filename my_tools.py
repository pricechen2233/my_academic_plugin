#my_tools
class PasswordCheck():
    '''
    密码校验工具。
    初始化方法：password_check=PasswordCheck("数据库名")。
    密码检查方法：.check("用户名","密码")，返回布尔类型。
    '''
    from my_fake_db_connect import fake_db_connect#数据库连接模块
    from passlib.context import CryptContext#哈希算法模块
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")#定义一个哈希算法


    def __init__(self,db:fake_db_connect):#初始化
        self.db = db


    def check(self,username:str,password:str)->bool:
        try:
            userinfo=self.db.get(username)#尝试取数据
            hashed_password=userinfo.get('hashed_password')
            if self.pwd_context.verify(password, hashed_password):
                return True
            else: return False
        except:
            return False
      

    def __call__(self,username:str,password:str)->bool:
        self.check(username, password)


class TokenCreator():
    '''
    令牌生成器。
    '''
    #算法库
    from jose import JWTError, jwt

    from datetime import datetime, timedelta
    from typing import Optional

    #令牌加密方法和有效期
    SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60


    def __init__(self):
        pass
    
    def __call__(self,username:str):
        self.create(username)

    def create(self,username:str,expires_delta:Optional[timedelta] = None)->str:
        if expires_delta:
            # 失效时间等于当前时间加有效期
            expire = self.datetime.utcnow()+expires_delta
        else:
            expire = self.datetime.utcnow()+self.timedelta(minutes=15)

        data={"sub":username,'exp':expire}#写入令牌的数据
        encode_jwt = self.jwt.encode(data, self.SECRET_KEY, algorithm=self.ALGORITHM)#生成令牌
        return encode_jwt

    def decode(self,token:str)->str:
        '''
        解码Token,返回用户名
        '''
        try:
            payload = self.jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return None

        except self.JWTError:
            return None
            print('解码令牌出错')
        return username

