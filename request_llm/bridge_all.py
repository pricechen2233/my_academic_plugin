
"""
    该文件中主要包含2个函数，是所有LLM的通用接口，它们会继续向下调用更底层的LLM模型，处理多模型并行等细节

    不具备多线程能力的函数：正常对话时使用，具备完备的交互功能，不可多线程
    1. predict(...)

    具备多线程调用能力的函数：在函数插件中被调用，灵活而简洁
    2. predict_no_ui_long_connection(...)
"""
import tiktoken
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from toolbox import get_conf, trimmed_format_exc

from .bridge_chatgpt import predict_no_ui_long_connection as chatgpt_noui
from .bridge_chatgpt import predict as chatgpt_ui

from .bridge_chatglm import predict_no_ui_long_connection as chatglm_noui
from .bridge_chatglm import predict as chatglm_ui

from .bridge_newbing import predict_no_ui_long_connection as newbing_noui
from .bridge_newbing import predict as newbing_ui

# from .bridge_tgui import predict_no_ui_long_connection as tgui_noui
# from .bridge_tgui import predict as tgui_ui

colors = ['#FF00FF', '#00FFFF', '#FF0000', '#990099', '#009999', '#990044']

class LazyloadTiktoken(object):
    def __init__(self, model):
        self.model = model

    @staticmethod
    @lru_cache(maxsize=128)
    def get_encoder(model):
        print('正在加载tokenizer，如果是第一次运行，可能需要一点时间下载参数')
        tmp = tiktoken.encoding_for_model(model)
        print('加载tokenizer完毕')
        return tmp
    
    def encode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model) 
        return encoder.encode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model) 
        return encoder.decode(*args, **kwargs)

# Endpoint 重定向
API_URL_REDIRECT, = get_conf("API_URL_REDIRECT")
openai_endpoint = "https://api.openai.com/v1/chat/completions"
api2d_endpoint = "https://openai.api2d.net/v1/chat/completions"
newbing_endpoint = "wss://sydney.bing.com/sydney/ChatHub"
# 兼容旧版的配置
try:
    API_URL, = get_conf("API_URL")
    if API_URL != "https://api.openai.com/v1/chat/completions": 
        openai_endpoint = API_URL
        print("警告！API_URL配置选项将被弃用，请更换为API_URL_REDIRECT配置")
except:
    pass
# 新版配置
if openai_endpoint in API_URL_REDIRECT: openai_endpoint = API_URL_REDIRECT[openai_endpoint]
if api2d_endpoint in API_URL_REDIRECT: api2d_endpoint = API_URL_REDIRECT[api2d_endpoint]
if newbing_endpoint in API_URL_REDIRECT: newbing_endpoint = API_URL_REDIRECT[newbing_endpoint]


# 获取tokenizer
tokenizer_gpt35 = LazyloadTiktoken("gpt-3.5-turbo")
tokenizer_gpt35_16k = LazyloadTiktoken("gpt-3.5-turbo-16k")
tokenizer_gpt4 = LazyloadTiktoken("gpt-4")
get_token_num_gpt35 = lambda txt: len(tokenizer_gpt35.encode(txt, disallowed_special=()))
get_token_num_gpt35_16k = lambda txt: len(tokenizer_gpt35_16k.encode(txt, disallowed_special=()))
get_token_num_gpt4 = lambda txt: len(tokenizer_gpt4.encode(txt, disallowed_special=()))


model_info = {
    # openai
    "gpt-3.5-turbo": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    #新增16k模型
    "gpt-3.5-turbo-16k": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 16384,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35_16k,
    },

    "gpt-4": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

    # api_2d
    "api2d-gpt-3.5-turbo": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": api2d_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    "api2d-gpt-4": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": api2d_endpoint,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

    # chatglm
    "chatglm": {
        "fn_with_ui": chatglm_ui,
        "fn_without_ui": chatglm_noui,
        "endpoint": None,
        "max_token": 1024,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    # newbing
    "newbing": {
        "fn_with_ui": newbing_ui,
        "fn_without_ui": newbing_noui,
        "endpoint": newbing_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

}


AVAIL_LLM_MODELS, = get_conf("AVAIL_LLM_MODELS")
if "jittorllms_rwkv" in AVAIL_LLM_MODELS:
    from .bridge_jittorllms_rwkv import predict_no_ui_long_connection as rwkv_noui
    from .bridge_jittorllms_rwkv import predict as rwkv_ui
    model_info.update({
        "jittorllms_rwkv": {
            "fn_with_ui": rwkv_ui,
            "fn_without_ui": rwkv_noui,
            "endpoint": None,
            "max_token": 1024,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "jittorllms_llama" in AVAIL_LLM_MODELS:
    from .bridge_jittorllms_llama import predict_no_ui_long_connection as llama_noui
    from .bridge_jittorllms_llama import predict as llama_ui
    model_info.update({
        "jittorllms_llama": {
            "fn_with_ui": llama_ui,
            "fn_without_ui": llama_noui,
            "endpoint": None,
            "max_token": 1024,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "jittorllms_pangualpha" in AVAIL_LLM_MODELS:
    from .bridge_jittorllms_pangualpha import predict_no_ui_long_connection as pangualpha_noui
    from .bridge_jittorllms_pangualpha import predict as pangualpha_ui
    model_info.update({
        "jittorllms_pangualpha": {
            "fn_with_ui": pangualpha_ui,
            "fn_without_ui": pangualpha_noui,
            "endpoint": None,
            "max_token": 1024,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "moss" in AVAIL_LLM_MODELS:
    from .bridge_moss import predict_no_ui_long_connection as moss_noui
    from .bridge_moss import predict as moss_ui
    model_info.update({
        "moss": {
            "fn_with_ui": moss_ui,
            "fn_without_ui": moss_noui,
            "endpoint": None,
            "max_token": 1024,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        },
    })
if "stack-claude" in AVAIL_LLM_MODELS:
    from .bridge_stackclaude import predict_no_ui_long_connection as claude_noui
    from .bridge_stackclaude import predict as claude_ui
    # claude
    model_info.update({
        "stack-claude": {
            "fn_with_ui": claude_ui,
            "fn_without_ui": claude_noui,
            "endpoint": None,
            "max_token": 8192,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        }
    })
if "newbing-free" in AVAIL_LLM_MODELS:
    try:
        from .bridge_newbingfree import predict_no_ui_long_connection as newbingfree_noui
        from .bridge_newbingfree import predict as newbingfree_ui
        # claude
        model_info.update({
            "newbing-free": {
                "fn_with_ui": newbingfree_ui,
                "fn_without_ui": newbingfree_noui,
                "endpoint": newbing_endpoint,
                "max_token": 4096,
                "tokenizer": tokenizer_gpt35,
                "token_cnt": get_token_num_gpt35,
            }
        })
    except:
        print(trimmed_format_exc())

def LLM_CATCH_EXCEPTION(f):
    """
    装饰器函数，将错误显示出来
    """
    def decorated(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience):
        try:
            return f(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
        except Exception as e:
            tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
            observe_window[0] = tb_str
            return tb_str
    return decorated


def predict_no_ui_long_connection(inputs, llm_kwargs, chatbot, history, sys_prompt, observe_window, console_slience=False):
    """
    发送至LLM，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
    inputs：
        是本次问询的输入
    sys_prompt:
        系统静默prompt
    llm_kwargs：
        LLM的内部调优参数
    history：
        是之前的对话列表
    observe_window = None：
        用于负责跨越线程传递已经输出的部分，大部分时候仅仅为了fancy的视觉效果，留空即可。observe_window[0]：观测窗。observe_window[1]：看门狗
    """
    import threading, time, copy

    model = llm_kwargs['llm_model']
    n_model = 1
    if '&' not in model:
        assert not model.startswith("tgui"), "TGUI不支持函数插件的实现"
        """
        # 可以插入计价程序
        # 记录用户输入的提示词数量并进行计价
        counter = model_info[llm_kwargs['llm_model']]["token_cnt"]
        prompt_tokens_count = counter(inputs)
        MODEL_PRICE, = get_conf("MODEL_PRICE")
        prompt_money = prompt_tokens_count*MODEL_PRICE[llm_kwargs['llm_model']]['prompt_price'] # 调用api输入的成本
        print("本次prompt成本为：", prompt_money)

        token = chatbot.get_cookies()['access_token']
        # 尝试解码token获取username
        username=tokencreator.decode(token)  
        if username is None: raise gr.Error("You are not login, please login!")

        # 尝试从数据库取用户信息
        try: userinfo=db.get(username)
        except: raise gr.Error("You are not login, please login!")
        
        if userinfo is None: raise gr.Error("You are not login, please login!")
        print(userinfo)
        # 检验是否有足够的余额，当余额少于足够支付1000prompt和1000completion时停止服务
        if(userinfo['money']-prompt_money <= 0.005):
            raise gr.Error("You don't have enough money!")
        # 计算收费并修改数据库
        #根据用户等级计算收费标准
        if userinfo["grade"] == 1 or userinfo["grade"] == 2:
            income = str(prompt_money*12/7)
            income = Decimal(income).quantize(Decimal('0.00000'),rounding=ROUND_HALF_UP)
            income = float(income)
            print("本次收费为：", income)
        else:
            income = str(prompt_money*20/7)
            income = Decimal(income).quantize(Decimal('0.00000'),rounding=ROUND_HALF_UP)
            income = float(income)
            print("本次收费为：", income)

        # 写入数据库
        from .io_database import datarw
        datarw.writeinfo(once_token=token,spend=income)
        """
        # 如果只询问1个大语言模型：
        method = model_info[model]["fn_without_ui"]
        return method(inputs, llm_kwargs, chatbot, history, sys_prompt, observe_window, console_slience)
    else:
        # 如果同时询问多个大语言模型：
        executor = ThreadPoolExecutor(max_workers=4)
        models = model.split('&')
        n_model = len(models)
        
        window_len = len(observe_window)
        assert window_len==3
        window_mutex = [["", time.time(), ""] for _ in range(n_model)] + [True]

        futures = []
        for i in range(n_model):
            model = models[i]
            method = model_info[model]["fn_without_ui"]
            llm_kwargs_feedin = copy.deepcopy(llm_kwargs)
            llm_kwargs_feedin['llm_model'] = model
            future = executor.submit(LLM_CATCH_EXCEPTION(method), inputs, llm_kwargs_feedin, chatbot, history, sys_prompt, window_mutex[i], console_slience)
            futures.append(future)

        def mutex_manager(window_mutex, observe_window):
            while True:
                time.sleep(0.25)
                if not window_mutex[-1]: break
                # 看门狗（watchdog）
                for i in range(n_model): 
                    window_mutex[i][1] = observe_window[1]
                # 观察窗（window）
                chat_string = []
                for i in range(n_model):
                    chat_string.append( f"【{str(models[i])} 说】: <font color=\"{colors[i]}\"> {window_mutex[i][0]} </font>" )
                res = '<br/><br/>\n\n---\n\n'.join(chat_string)
                # # # # # # # # # # #
                observe_window[0] = res

        t_model = threading.Thread(target=mutex_manager, args=(window_mutex, observe_window), daemon=True)
        t_model.start()

        return_string_collect = []
        while True:
            worker_done = [h.done() for h in futures]
            if all(worker_done):
                executor.shutdown()
                break
            time.sleep(1)

        for i, future in enumerate(futures):  # wait and get
            return_string_collect.append( f"【{str(models[i])} 说】: <font color=\"{colors[i]}\"> {future.result()} </font>" )

        window_mutex[-1] = False # stop mutex thread
        res = '<br/><br/>\n\n---\n\n'.join(return_string_collect)
        return res
# predict 函数需要的模块
import gradio as gr
from main import tokencreator,db
from decimal import *

def predict(inputs, llm_kwargs, *args, **kwargs):
    """
    发送至LLM，流式获取输出。
    用于基础的对话功能。
    inputs 是本次问询的输入
    top_p, temperature是LLM的内部调优参数
    history 是之前的对话列表（注意无论是inputs还是history，内容太长了都会触发token数量溢出的错误）
    chatbot 为WebUI中显示的对话列表，修改它，然后yeild出去，可以直接修改对话界面内容
    additional_fn代表点击的哪个按钮，按钮见functional.py
    """
    """
    # 记录用户输入的提示词数量并进行计价
    counter = model_info[llm_kwargs['llm_model']]["token_cnt"]
    prompt_tokens_count = counter(inputs)
    MODEL_PRICE, = get_conf("MODEL_PRICE")
    prompt_money = prompt_tokens_count*MODEL_PRICE[llm_kwargs['llm_model']]['prompt_price'] # 调用api输入的成本
    print("本次的prompt成本为：", prompt_money)

    res = []
    for arg in args:
        res.append(arg)
    # 从chatbot_with_cookie得到access token
    token = res[1].get_cookies()['access_token']
    print(token)
    # 尝试解码token获取username
    username=tokencreator.decode(token)  
    if username is None: raise gr.Error("You are not login, please login!")

	# 尝试从数据库取用户信息
    try: userinfo=db.get(username)
    except: raise gr.Error("You are not login, please login!")
       
    if userinfo is None: raise gr.Error("You are not login, please login!")
    print(userinfo)
    # 检验是否有足够的余额，当余额少于足够支付1000prompt和1000completion时停止服务
    if(userinfo['money']-prompt_money <= 0.005):
        raise gr.Error("You don't have enough money!")
    # 计算收费并修改数据库
    #根据用户等级计算收费标准
    if userinfo["grade"] == 1 or userinfo["grade"] == 2:
        income = str(prompt_money*12/7)
        income = Decimal(income).quantize(Decimal('0.00000'),rounding=ROUND_HALF_UP)
        income = float(income)
        print("本次收费为：", income)
    else:
        income = str(prompt_money*20/7)
        income = Decimal(income).quantize(Decimal('0.00000'),rounding=ROUND_HALF_UP)
        income = float(income)
        print("本次收费为：", income)

    # 写入数据库
    from .io_database import datarw
    datarw.writeinfo(once_token=token,spend=income)
    """
    # 以下为向LLM发送信息的模块
    method = model_info[llm_kwargs['llm_model']]["fn_with_ui"]
    yield from method(inputs, llm_kwargs, *args, **kwargs)

