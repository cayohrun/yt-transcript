# yt-transcript skill

這個 repo 以 **Agent Skills** 標準提供 `yt-transcript` 技能，核心內容在：

- `skills/yt-transcript/SKILL.md`

skill 內含可執行腳本：`scripts/get_transcripts.py`（與 repo 根目錄版本一致）。

## 安裝（四平台）

> 下列路徑擇一即可。建議用 **copy 或 symlink** 方式把 skill 資料夾放進對應位置。

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

## 需求

- Python 3
- 套件：`pandas`、`requests`

## 使用方式（技能內容）

- 技能內容在：`skills/yt-transcript/SKILL.md`
- 執行腳本時，請在「要輸出結果的資料夾」執行：

```bash
./.venv/bin/python <skill-root>/scripts/get_transcripts.py
```

`<skill-root>` 指的是包含 `SKILL.md` 的資料夾。

若在 repo 內執行，可用：

```bash
./.venv/bin/python ./skills/yt-transcript/scripts/get_transcripts.py
```

### API Key（不要用到他人的 key）

- 用環境變數：

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

- 或在本機放 key 檔（不提交）：  
  - `.gemini_key_paid` / `.gemini_key_free`（放在 `scripts/` 同一層）

`.gemini_key*` 已在 `.gitignore`，不要 commit。

## 直接貼 URL（單支/多支）

```bash
SOURCE_NAME="LLM" URLS="https://youtu.be/AAA https://youtu.be/BBB" \
  ./.venv/bin/python <skill-root>/scripts/get_transcripts.py
```

## 打包 `.skill`

已提供 `.skill` 發佈檔（含 `scripts/`）：

- `dist/yt-transcript.skill`
- `yt-transcript.skill`（同一份副本，方便直接下載）

如需自行重新打包，可使用 Agent Skills 參考工具：

```bash
python3 ~/.codex/skills/.system/skill-creator/scripts/package_skill.py \
  /Users/jw-mba/Projects/Active/yt-transcript/skills/yt-transcript \
  /Users/jw-mba/Projects/Active/yt-transcript/dist
```

## 安裝 `.skill`（範例）

```bash
# Codex（全域）
mkdir -p ~/.codex/skills
unzip -q yt-transcript.skill -d ~/.codex/skills

# Cursor（專案）
mkdir -p .cursor/skills
unzip -q yt-transcript.skill -d .cursor/skills

# Claude Code（專案）
mkdir -p .claude/skills
unzip -q yt-transcript.skill -d .claude/skills

# Antigravity（專案）
mkdir -p .agent/skills
unzip -q yt-transcript.skill -d .agent/skills
```

## License

MIT
