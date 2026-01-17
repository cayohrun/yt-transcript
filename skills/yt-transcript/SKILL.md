---
name: yt-transcript
description: 產生 YouTube 逐字稿並依 NotebookLM 來源上限自動合併/分檔，支援 CSV 與直接 URL，包含續跑/補跑與重試。適用於要把多支 YouTube 影片整理成 NotebookLM 來源檔的場景。
license: MIT. LICENSE.txt has complete terms
---

# yt-transcript

## 目的

把 YouTube 影片逐字稿寫進 `transcript/<NotebookSource>.md`，若接近 NotebookLM 單一來源上限就自動續接 `_partN`。

## 前置條件

- 需要 Python 3 + 套件 `pandas`、`requests`
- 腳本位於本 skill 內：`scripts/get_transcripts.py`
- 腳本會在「目前工作目錄」建立 `transcript/`、`logs/` 等資料夾
- 只支援公開影片（private / unlisted 不可用）

完整設定、環境變數與進階用法請看 `reference.md`。

## 安全重點（不要用到使用者的 key）

- **不要把 API key 寫進檔案或提交到 Git**
- 請使用環境變數或本機 key 檔：
  - `GEMINI_API_KEY=...`（環境變數）
  - `.gemini_key_paid` / `.gemini_key_free`（放在 `scripts/` 同一層）

## 最短流程（CSV 模式）

`<skill-root>` 指的是包含本 `SKILL.md` 的資料夾。

1. `cd` 到要輸出結果的資料夾（包含 `youtube_videos.csv`）  
2. 執行：  
   - `./.venv/bin/python <skill-root>/scripts/get_transcripts.py`

## 最短流程（直接貼 URL）

- `SOURCE_NAME="LLM" URLS="https://youtu.be/AAA https://youtu.be/BBB" ./.venv/bin/python <skill-root>/scripts/get_transcripts.py`

## 輸入規則

- CSV 模式：
  - 欄位最少 `Title`、`URL`、`NotebookSource`
  - 腳本會自動新增 `Checked` 與 `序號`
- URL 模式：
  - 必須指定 `SOURCE_NAME`
  - `URLS`（多支可空白分隔）或 `URLS_FILE`（每行一支）

## 輸出規則

- 來源檔：`transcript/<NotebookSource>.md`
- 滿上限續接：`transcript/<NotebookSource>_partN.md`
- 上限計算：`NOTEBOOKLM_MAX_WORDS × NOTEBOOKLM_TARGET_RATIO`（預設 500000 × 0.8）

## 常用開關

- `MAX_VIDEOS=10`：只跑 N 支
- `RETRY_INDEX=82`：單支補跑（CSV 序號）
- `MERGE_LINES=1`：合併短行
- `GEMINI_MODEL`：指定模型

## 檢查與續跑

- 已寫入的 URL 會自動略過
- `Checked` 欄位：`✔` 正常、`△` MAX_TOKENS、`✖` 失敗
- 可用 `RETRY_INDEX` 單支補跑
