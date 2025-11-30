import os
import requests
import json
import time
from datetime import datetime, timedelta
import concurrent.futures
from bs4 import BeautifulSoup, Comment
from openai import OpenAI
from urllib.parse import urljoin

# ================= 配置区域 =================
# 请确保环境变量中设置了 OPENAI_API_KEY
API_KEY = os.getenv("OPENAI_API_KEY")
# 如果你使用的是中转或者自定义端点，请修改 base_url
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/")

MODEL_NAME = os.getenv("MODEL_NAME", "qwen-flash")

 

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 请求头，模拟浏览器防止被简单拦截
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ================= 工具函数 =================

def fetch_page_content(url):
    """
    获取网页HTML内容，并进行简单的清洗（去除JS和CSS），
    以减少传给LLM的Token数量。
    """
    try:
        print(f"正在获取页面: {url} ...")
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        # 使用 BeautifulSoup 清洗无关标签
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 移除 style, noscript, footer, header, nav 等通常不包含核心内容的标签
        # 注意：保留 meta 和 script 以便进一步提取时间信息
        # 新增移除: svg, img, link, iframe, button, input, form (大幅减少无关数据)
        for tag in soup(["style", "noscript", "footer", "header", "nav", "iframe", "svg", "img", "link", "button", "input", "form"]):
            tag.extract()

        # 移除注释
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
            
        # 筛选 script 标签：仅保留包含时间关键词的脚本 (例如 JSON-LD 或 变量定义)
        for script in soup(["script"]):
            # 如果是外部脚本(有src)或没有内容，通常不包含我们要的直接时间数据，去除
            if not script.string:
                script.extract()
                continue
                
            content = script.string.lower()
            # 关键词列表
            if any(kw in content for kw in ["timestamp", "datetime", "pubdate", "published_time"]):
                continue # 保留
            
            script.extract()
            
        # 获取纯文本或保留部分HTML结构。
        # 为了保留 head 中的 meta 和 script，我们需要返回整个 soup (或者 head + body)
        clean_html = str(soup) # 截取前15000字符，视模型上下文窗口而定
        return clean_html
    except Exception as e:
        print(f"获取页面失败: {e}")
        return None

def fetch_article_content(url):
    """
    获取文章详情页的正文内容（纯文本）。
    相比 fetch_page_content，此函数更专注于提取文章核心文本，
    去除导航、广告、侧边栏等无关干扰。
    """
    try:
        print(f"正在获取文章详情: {url} ...")
        response = requests.get(url, headers=HEADERS, timeout=100)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. 移除无关标签
        # 移除 script, style, noscript, footer, header, nav, iframe, svg, img, link, button, input, form, aside, meta
        for tag in soup(["script", "style", "noscript", "footer", "header", 
                         "nav", "iframe", "svg", "img", "link", "button", 
                         "input", "form", "aside", "meta"]):
            tag.extract()
            
        # 移除注释
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
            
        # 2. 尝试提取正文
        # 策略：寻找 <article> 或 特定 class 的 div
        content_node = soup.find('article')
        
        if not content_node:
            # 尝试查找常见的正文容器 class
            possible_classes = ['article', 'content', 'post-content', 'news-body', 'story-body', 'main']
            for cls in possible_classes:
                # 模糊匹配 class
                content_node = soup.find('div', class_=lambda x: x and cls in x)
                if content_node:
                    break
        
        # 3. 获取文本
        if content_node:
            text = content_node.get_text(separator="\n", strip=True)
        else:
            # 兜底：提取所有 P 标签
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
            # 过滤掉过短的段落（可能是导航项）
            paragraphs = [p for p in paragraphs if len(p) > 10]
            text = "\n\n".join(paragraphs)
            
            # 如果 P 标签也很少，最后尝试 body
            if len(text) < 50 and soup.body:
                text = soup.body.get_text(separator="\n", strip=True)
        
        return text
        
    except Exception as e:
        print(f"获取文章详情失败: {e}")
        return None

def extract_article_list_with_llm(html_content, base_url):
    """
    使用 LLM 分析主页 HTML，提取文章列表。
    """
    print("正在使用 LLM 解析主页结构...")
    
    prompt = f"""
    你是一个智能爬虫助手。我提供了一个新闻网站主页的HTML片段（已清洗）。
    请从中提取新闻文章列表。
    
    网站 Base URL 是: {base_url}
    
    要求：
    1. 提取每篇文章的标题 (title)。
    2. 提取完整的文章链接 (url)。如果是相对路径，请基于 Base URL 补全。
    3. 提取发布时间。
       - 如果是日期时间文本，请转换为 "YYYY-MM-DD HH:MM:SS" 格式，存入 "datetime_str" 字段。
       - 如果是时间戳（数字），请存入 "timestamp" 字段。
       - 如果找不到具体时间，请留空。
    
    输出格式必须是纯 JSON 对象，格式如下：
    {{
        "articles": [
            {{
                "title": "文章标题",
                "url": "https://example.com/article/1",
                "datetime_str": "2023-10-01 12:00:00",
                "timestamp": 1696161600
            }}
        ]
    }}
    
    HTML 片段如下:
    {html_content}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful data extraction assistant that outputs strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("articles", [])
    except Exception as e:
        print(f"LLM 提取列表失败: {e}")
        return []

def summarize_article_with_llm(content):
    """
    使用 LLM 总结文章内容。
    """
    prompt = f"""
    请阅读以下新闻网页的原始文本，并生成一个简洁的中文摘要（100字以内）。
    
    网页内容:
    {content[:5000]} 
    """
    # 注意：截取前5000字符防止Token溢出

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful news summarizer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"摘要生成失败: {e}"

def parse_input_time(time_str):
    """
    解析输入的时间字符串 "YYYY-MM-DD HH:MM:SS"
    """
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            # 兼容旧格式
            return datetime.strptime(time_str, "%Y%m%d %H%M%S")
        except ValueError:
            print("时间格式错误，请使用 YYYY-MM-DD HH:MM:SS")
            return None

def process_single_article(article, homepage_url, start_time, end_time, enable_summary, return_content=False):
    """
    处理单篇文章的逻辑，用于并行执行
    """
    title = article.get("title")
    url = article.get("url")
    time_str = article.get("datetime_str")
    timestamp = article.get("timestamp")
    
    # 如果 URL 是相对路径，进行补全
    if url and not url.startswith("http"):
        url = urljoin(homepage_url, url)

    # 时间过滤逻辑
    should_process = False
    article_time = None

    # 优先尝试解析 datetime_str
    if time_str:
        try:
            article_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(f"警告(时间字符串解析失败): {time_str} - {title}")
    
    # 如果没有 datetime_str 或解析失败，尝试使用 timestamp
    if not article_time and timestamp:
        try:
            ts = float(timestamp)
            # 简单的毫秒/秒判断
            if ts > 30000000000: 
                ts = ts / 1000
            article_time = datetime.fromtimestamp(ts)
            # 更新 time_str 以便后续显示
            time_str = article_time.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            print(f"警告(时间戳解析失败): {timestamp} - {title}")

    if article_time:
        if start_time <= article_time <= end_time:
            should_process = True
        else:
            print(f"跳过(不在时间范围内): {title} ({time_str})")
    else:
        print(f"跳过(无时间): {title}")
        return None

    if should_process:
        print(f"正在处理: {title}...")
        
        summary = ""
        content = ""
        if enable_summary or return_content:
            # 获取文章详情页
            article_content = fetch_article_content(url)
            if article_content:
                if enable_summary:
                    # LLM 摘要
                    summary = summarize_article_with_llm(article_content)
                if return_content:
                    content = article_content
        
        return {
            "title": title,
            "summary": summary,
            "content": content,
            "url": url,
            "datetime": time_str
        }
    
    return None

# ================= 主逻辑 =================

def process_news(homepage_url, start_time_str, end_time_str, enable_summary=True, return_content=False):
    """
    主流程函数
    :param homepage_url: 新闻主页 URL
    :param start_time_str: 开始时间字符串 'YYYY-MM-DD HH:MM:SS'
    :param end_time_str: 结束时间字符串 'YYYY-MM-DD HH:MM:SS'
    :param enable_summary: 是否生成文章摘要
    """
    
    # 1. 解析目标时间
    start_time = parse_input_time(start_time_str)
    end_time = parse_input_time(end_time_str)
    
    if not start_time or not end_time:
        print("时间解析失败，无法继续")
        return

    # 2. 获取主页内容
    homepage_html = fetch_page_content(homepage_url)
    if not homepage_html:
        return

    # 3. LLM 提取文章列表
    articles = extract_article_list_with_llm(homepage_html, homepage_url)
    print(f"LLM 提取到 {len(articles)} 篇文章，开始按时间过滤...")

    final_results = []

    # 4. 并行处理
    # 使用 ThreadPoolExecutor 并行执行
    # 线程数建议不要太多，以免触发反爬限制或 API 速率限制
    max_workers = 5 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_article = {
            executor.submit(process_single_article, article, homepage_url, start_time, end_time, enable_summary, return_content): article  
            for article in articles
        }
        
        for future in concurrent.futures.as_completed(future_to_article):
            try:
                result = future.result()
                if result:
                    final_results.append(result)
            except Exception as exc:
                print(f"处理文章时发生异常: {exc}")

    return final_results

# if __name__ == "__main__":
#     # ================= 使用示例 =================
    
#     # 参数设置
#     # 注意：请替换为一个实际的新闻列表页面 URL
#     TARGET_URL = "https://timesofindia.indiatimes.com/india"
#     # 或者尝试一个简单的博客列表页
    
#     # 设定时间范围
#     # 格式：YYYY-MM-DD HH:MM:SS
#     # 默认为当前时间
#     START_TIME = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S") 
#     END_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

#     print(f"开始任务: 抓取 {TARGET_URL} 上 {START_TIME} 到 {END_TIME} 之间的新闻...\n")

#     # enable_summary=True 开启摘要，False 只获取链接
#     results = process_news(TARGET_URL, START_TIME, END_TIME, enable_summary=True)

#     print("\n" + "="*30)
#     print(f"最终结果 (共 {len(results)} 条):")
#     print("="*30)
    
#     for item in results:
#         print(f"标题: {item['title']}")
#         print(f"时间: {item['datetime']}")
#         print(f"链接: {item['url']}")
#         print(f"摘要: {item['summary']}")
#         print("-" * 20)

#     # 你也可以将 results 保存为 CSV 或 JSON
#     with open("summary_report.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)


