---
name: yt-transcript
description: 產生 YouTube 逐字稿並依 NotebookLM 來源上限自動合併/分檔，支援 CSV 與直接 URL，包含續跑/補跑與重試。適用於要把多支 YouTube 影片整理成 NotebookLM 來源檔的場景。
license: MIT
---

# yt-transcript

## 目的

把 YouTube 影片逐字稿寫進 `transcript/<NotebookSource>.md`，若接近 NotebookLM 單一來源上限就自動續接 `_partN`。

## 前置條件

- 這個 skill **只提供流程與操作**，實際執行使用 repo 內的 `get_transcripts.py`
- 請先在 repo 內建好虛擬環境並安裝相依套件
- 只支援公開影片（private / unlisted 不可用）

## 安全重點（不要用到使用者的 key）

- **不要把 API key 寫進檔案或提交到 Git**
- 請使用環境變數或本機 key 檔：
  - `GEMINI_API_KEY=...`（環境變數）
  - `.gemini_key_paid` / `.gemini_key_free`（本機檔案，已在 `.gitignore`）

## 最短流程（CSV 模式）

1. 確認 `youtube_videos.csv` 有 `Title`、`URL`、`NotebookSource`
2. 在 repo 根目錄執行：
   - `./.venv/bin/python get_transcripts.py`

## 最短流程（直接貼 URL）

- `SOURCE_NAME="LLM" URLS="https://youtu.be/AAA https://youtu.be/BBB" ./.venv/bin/python get_transcripts.py`

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
