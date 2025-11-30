{
  "mcpServers": {
    "fetch-news": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/PuWenyin/fetch-news-mcp.git",
        "fetch-news" 
      ],
      "env": {
        "OPENAI_API_KEY": "sk-xxxxxxxxx"
      }
    }
  }
}