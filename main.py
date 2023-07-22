#文件名:main.py
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware#跨域请求
from fastapi.templating import Jinja2Templates#连接模板库
from fastapi.responses import HTMLResponse, RedirectResponse
from my_fake_db_connect import fake_db_connect#虚拟数据库连接
from my_tools import PasswordCheck, TokenCreator
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles#创建静态文件目录
from toolbox import get_conf

# 读取数据库各种信息
DATABASE_HOST, DATABASE_USER, DATABASE_PASSWD, DATABASE_NAME = get_conf("DATABASE_HOST", "DATABASE_USER", "DATABASE_PASSWD", "DATABASE_NAME")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")#初始化安全性方案对象

#初始化工具类 解码器和数据库
tokencreator=TokenCreator()
db=fake_db_connect(host=DATABASE_HOST,user=DATABASE_USER,password=DATABASE_PASSWD,database=DATABASE_NAME)
passwordcheck=PasswordCheck(db)

# 放行跨域请求的域名
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8848",
    "http://127.0.0.1:8848",
    "http://127.0.0.1:5500"
]

app=FastAPI()
templates = Jinja2Templates(directory="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 添加跨域方案
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 用户登录页面(首页)
@app.get('/', response_class=HTMLResponse)
async def show_login(request: Request):
    return templates.TemplateResponse('login.html', {'request': request})

#登录，返回令牌
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    '''
    处理登录请求
    '''
    #user = authenticate_user(
    #    fake_users_db, form_data.username, form_data.password)
    username=form_data.username
    password=form_data.password

    if not passwordcheck.check(username, password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token=tokencreator.create(username)
    print('返回用户令牌',access_token)
    # 登录成功，重定向到欢迎页面
    response = RedirectResponse(url='/welcome')
    # 可以在响应中设置cookie等信息
    response.set_cookie(key='access_token', value=access_token, httponly=True)
    return response

#用户欢迎页
@app.post("/welcome")
async def show_welcome(request: Request):
    # 检查用户是否登录
    if not request.cookies.get('access_token'):
        # 未登录，重定向到登录页面
        return RedirectResponse(url='/')

    return templates.TemplateResponse('welcome.html', {'request': request})

# 用户欢迎页面
@app.get("/user/me")
async def get_current_user(request: Request):
    '''
    获取当前用户信息。
    '''
    # 默认权限错误信息
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token = request.cookies.get('access_token')
    print("传入用户令牌",token)
    # 尝试解码token获取username
    username=tokencreator.decode(token)
    
    if username is None: raise credentials_exception

	# 尝试从数据库取用户信息
    try: userinfo=db.get(username)
    except: raise credentials_exception
       
    if userinfo is None: raise credentials_exception
    print(userinfo)
    print(type(userinfo))

    return {'usernames':userinfo['username'], 'money':userinfo['money']}

from gptac import gptac
import gradio as gr

demo = gptac()
#有问题需要改，主要是将gptac所有的输入click函数的fn对应predict函数都加入身份验证
"""
def leadtogpt(request: gr.Request):
    if 'access_token' not in request.cookies.keys():
        raise gr.Error("You are not login, please login!")
    token = request.cookies['access_token']
    print(token)
    # 尝试解码token获取username
    username=tokencreator.decode(token)
    if username is None: raise gr.Error("You are not login, please login!")
	# 尝试从数据库取用户信息
    try: userinfo=db.get(username)
    except: raise gr.Error("You are not login, please login!")
    if userinfo is None: raise gr.Error("You are not login, please login!")
    print(userinfo)

demo.load(leadtogpt)"""
app = gr.mount_gradio_app(app,demo,path='/gpt')

