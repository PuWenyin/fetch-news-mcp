import concurrent.futures
from fastmcp import FastMCP
from main import process_news
from typing import Annotated
from pydantic import Field
from datetime import datetime, timedelta

mcp = FastMCP("fetch-news")

@mcp.tool(description="获取特定页面的某个时间范围的新闻内容，如果没有开启return_content，那么返回URL、发布时间、标题，如果开启return_content，那么返回URL、标题、发布时间、内容")
def fetch_news(
    urls: Annotated[list[str], Field(description="目标页面的URL列表")],
    start_time: Annotated[str, Field(description="起始时间，YYYY-MM-DD HH:MM:SS")] = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S") ,
    end_time: Annotated[str, Field(description="结束时间，YYYY-MM-DD HH:MM:SS")] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") ,
    return_content: Annotated[bool, Field(description="是否返回正文内容")] = True,
):
    """
    并行抓取多个新闻网站主页的新闻内容。
    """
    all_results = []
    
    # 定义处理单个 URL 的函数
    def process_single_url(target_url):
        try:
            print(f"开始处理: {target_url}")
            # 调用 main.py 中的 process_news
            return process_news(target_url, start_time, end_time, False, return_content)
        except Exception as e:
            print(f"处理 {target_url} 失败: {e}")
            return []

    # 使用 ThreadPoolExecutor 并行处理多个 URL
    # 建议 worker 数量不要过多，避免并发过高
    max_workers = min(len(urls), 10) if urls else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(process_single_url, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
            except Exception as e:
                print(f"获取 {url} 结果时发生异常: {e}")

    return all_results

def main():
    """
    这是给 uvx (project.scripts) 调用的入口函数。
    它的作用等同于你在命令行运行 `fastmcp run mcp_app.py:mcp`
    """
    mcp.run()

if __name__ == "__main__":
    main()