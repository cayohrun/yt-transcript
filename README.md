# yt-transcript skill

把 YouTube 影片逐字稿整理成 NotebookLM 來源檔：同一個來源會自動合併，
接近來源上限就續接成 `_partN`，用來減少 NotebookLM 來源數量。

skill 內含可攜腳本：`scripts/get_transcripts.py`。

## 完整教學（一步一步）

`<skill-root>` 指包含 `SKILL.md` 的資料夾；在這個 repo 內是
`skills/yt-transcript`。

### 1) 準備環境

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install pandas requests
```

### 2) 設定 API key（不要用到他人的 key）

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

或放在 `<skill-root>/scripts/` 同層：

- `.gemini_key_paid`
- `.gemini_key_free`

### 3) 準備輸入

**CSV 模式**（`youtube_videos.csv`）：
必須有 `Title`、`URL`、`NotebookSource`

**URL 模式**：
直接提供 `SOURCE_NAME` + URL 清單

### 4) 執行

在「要輸出結果的資料夾」執行：

```bash
./.venv/bin/python <skill-root>/scripts/get_transcripts.py
```

URL 模式範例：

```bash
SOURCE_NAME="LLM" URLS="https://youtu.be/AAA https://youtu.be/BBB" \
  ./.venv/bin/python <skill-root>/scripts/get_transcripts.py
```

### 5) 產出

- 逐字稿：`transcript/<NotebookSource>.md` / `_partN.md`
- 紀錄：`logs/run_log.csv`

## 安裝（四平台）

> 下列路徑擇一即可。建議用 copy 或 symlink 把 `skills/yt-transcript/`
> 放進對應位置。

### Codex (CLI)

- 全域：`~/.codex/skills/yt-transcript/`

### Cursor

- 專案：`<repo>/.cursor/skills/yt-transcript/`
- 全域：`~/.cursor/skills/yt-transcript/`
- 也會掃描：`.claude/skills/` / `~/.claude/skills/`

### Claude Code

- 專案：`<repo>/.claude/skills/yt-transcript/`
- 全域：`~/.claude/skills/yt-transcript/`

### Antigravity

- 專案：`<workspace>/.agent/skills/yt-transcript/`
- 全域：`~/.gemini/antigravity/skills/yt-transcript/`

## Repo 結構

- `skills/yt-transcript/SKILL.md`
- `skills/yt-transcript/reference.md`
- `skills/yt-transcript/scripts/get_transcripts.py`
- `skills/yt-transcript/scripts/prompt.txt`
- `skills/yt-transcript/LICENSE.txt`

## License

MIT
