import csv
import os
import sys
import time
from datetime import datetime
import signal

import pandas as pd
import requests

STOP_REQUESTED = False


def _handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
GEMINI_MAX_CONTINUATIONS = int(os.getenv("GEMINI_MAX_CONTINUATIONS", "3"))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "120"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
EMPTY_RESPONSE_MAX_RETRIES = int(os.getenv("EMPTY_RESPONSE_MAX_RETRIES", "30"))
RETRY_FOREVER_ON_TIMEOUT = os.getenv("RETRY_FOREVER_ON_TIMEOUT", "1") == "1"
SLEEP_BETWEEN_CALLS = float(os.getenv("GEMINI_SLEEP", "0.6"))
OVERWRITE = os.getenv("OVERWRITE", "0") == "1"
MAX_VIDEOS = os.getenv("MAX_VIDEOS")
CHECK_COLUMN = os.getenv("CHECK_COLUMN", "Checked")
KEY_PROFILE_RAW = os.getenv("GEMINI_KEY_PROFILE", "").strip().lower()
KEY_PROFILE = KEY_PROFILE_RAW
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "run_log.csv")
INPUT_PRICE_PER_M = os.getenv("INPUT_PRICE_PER_M", "0.10")  # 2026-01-15
OUTPUT_PRICE_PER_M = os.getenv("OUTPUT_PRICE_PER_M", "0.40")  # 2026-01-15
GEMINI_MEDIA_RESOLUTION_RAW = os.getenv("GEMINI_MEDIA_RESOLUTION", "")
PROMPT_FILE = os.getenv("PROMPT_FILE", "prompt.txt")
MERGE_LINES = os.getenv("MERGE_LINES", "1") == "1"
RETRY_INDEX_RAW = os.getenv("RETRY_INDEX", "").strip()


class FreeTierLimitError(RuntimeError):
    pass


class EmptyResponseError(RuntimeError):
    def __init__(self, message, attempts=0, timeout_retries=0):
        super().__init__(message)
        self.attempts = attempts
        self.timeout_retries = timeout_retries


def _read_key(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            return key or None
    except FileNotFoundError:
        return None


if not GEMINI_API_KEY:
    base_dir = os.path.dirname(__file__)
    if KEY_PROFILE:
        key_path = os.path.join(base_dir, f".gemini_key_{KEY_PROFILE}")
        GEMINI_API_KEY = _read_key(key_path)
    else:
        candidates = [
            ("paid", ".gemini_key_paid"),
            ("default", ".gemini_key"),
            ("free", ".gemini_key_free"),
        ]
        for name, filename in candidates:
            key = _read_key(os.path.join(base_dir, filename))
            if key:
                GEMINI_API_KEY = key
                KEY_PROFILE = name
                break

if not KEY_PROFILE:
    if GEMINI_API_KEY:
        KEY_PROFILE = "env"
    else:
        KEY_PROFILE = "default"

DEFAULT_PROMPT = (
    "請產出該 YouTube 影片的完整逐字稿。\n"
    "要求：\n"
    "1) 不要摘要或省略任何內容。\n"
    "2) 只輸出逐字稿內容，不要加標題、說明或格式標記。\n"
    "3) 若是中文語音，請用繁體中文；非中文則保留原語。\n"
    "4) 若影片無語音，請回覆：[無語音]\n"
)


def _load_prompt():
    if not PROMPT_FILE:
        return DEFAULT_PROMPT
    path = PROMPT_FILE
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content or DEFAULT_PROMPT
    except FileNotFoundError:
        return DEFAULT_PROMPT


BASE_PROMPT = _load_prompt()


def _normalize_media_resolution(value):
    if not value:
        return ""
    v = value.strip().upper()
    if v in {"LOW", "MEDIA_RESOLUTION_LOW"}:
        return "MEDIA_RESOLUTION_LOW"
    if v in {"MEDIUM", "MEDIA_RESOLUTION_MEDIUM"}:
        return "MEDIA_RESOLUTION_MEDIUM"
    if v in {"HIGH", "MEDIA_RESOLUTION_HIGH"}:
        return "MEDIA_RESOLUTION_HIGH"
    if v in {"ULTRA_HIGH", "MEDIA_RESOLUTION_ULTRA_HIGH"}:
        return "MEDIA_RESOLUTION_ULTRA_HIGH"
    return value.strip()


GEMINI_MEDIA_RESOLUTION = _normalize_media_resolution(GEMINI_MEDIA_RESOLUTION_RAW)

LOG_HEADERS = [
    "工作階段ID",
    "層級",
    "開始時間",
    "結束時間",
    "耗時(秒)",
    "影片序號",
    "CSV行號",
    "影片標題",
    "影片連結",
    "寫入檔名",
    "完成狀態",
    "使用Key類型",
    "中斷原因",
    "錯誤訊息",
    "輸入Token數",
    "輸出Token數",
    "總Token數",
    "預估成本(USD)",
    "API呼叫次數",
    "續寫次數",
    "重試次數",
    "Timeout重試次數",
    "Finish原因",
    "模型名稱",
    "媒體解析度",
    "最大輸出Token",
    "超時秒數",
    "每次間隔秒數",
    "本次成功影片數",
    "本次失敗影片數",
    "本次總輸入Token",
    "本次總輸出Token",
    "本次總Token",
    "本次總預估成本(USD)",
]


def _parse_price(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_cost(input_tokens, output_tokens):
    input_price = _parse_price(INPUT_PRICE_PER_M)
    output_price = _parse_price(OUTPUT_PRICE_PER_M)
    if input_price is None or output_price is None:
        return None
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _write_log_row(log_path, row):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_rate_limit_error(message, status_code, status):
    msg = (message or "").lower()
    status_upper = (status or "").upper()
    if status_code == 429:
        return True
    if status_upper in {"RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED"}:
        return True
    keywords = [
        "rate limit",
        "quota",
        "resource exhausted",
        "too many requests",
        "exceeded",
    ]
    return any(keyword in msg for keyword in keywords)


def _format_free_tier_message(message):
    if message:
        return f"免費版已達請求上限，腳本已中斷。原因：{message}"
    return "免費版已達請求上限，腳本已中斷。"


def _merge_transcript_lines(text, soft_wrap=120, max_len=180):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    output = []
    buf = ""

    def is_heading(line):
        return line.endswith("：") or line.endswith(":")

    for line in lines:
        if is_heading(line):
            if buf:
                output.append(buf)
                buf = ""
            output.append(line)
            continue
        if not buf:
            buf = line
            continue
        if len(buf) >= soft_wrap and len(buf) + len(line) > max_len:
            output.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        output.append(buf)
    return "\n".join(output)


def _build_part_header(part_num, start_idx, end_idx):
    return (
        f"# 易學基地頻道 影片逐字稿 Part {part_num}\n"
        f"**範圍：影片 {start_idx:03d} - {end_idx:03d}**\n\n---\n\n"
    )


def _build_entry(index, title, url, transcript_text):
    formatted = "\n".join([f"    > {line}" for line in transcript_text.splitlines()])
    return (
        f"## {index + 1}. {title}\n"
        f"* **影片連結**: {url}\n"
        "* **逐字稿**:\n"
        f"{formatted}\n\n---\n\n"
    )


def _replace_entry_in_file(filename, header, url, entry):
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(entry)
        return "created"

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines(keepends=True)
    url_line = f"* **影片連結**: {url}"
    url_idx = None
    for i, line in enumerate(lines):
        if line.strip() == url_line:
            url_idx = i
            break

    if url_idx is None:
        if not content.endswith("\n"):
            content += "\n"
        content += entry
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return "appended"

    start_idx = 0
    for i in range(url_idx, -1, -1):
        if lines[i].startswith("## "):
            start_idx = i
            break

    end_idx = len(lines)
    for i in range(url_idx, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i + 1
            if end_idx < len(lines) and not lines[end_idx].strip():
                end_idx += 1
            break

    new_lines = lines[:start_idx] + entry.splitlines(keepends=True) + lines[end_idx:]
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    return "replaced"


def _call_gemini(session, url, prompt):
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    file_part = {"file_data": {"file_uri": url}}
    if GEMINI_MEDIA_RESOLUTION:
        file_part["mediaResolution"] = GEMINI_MEDIA_RESOLUTION
    payload = {
        "contents": [
            {
                "parts": [
                    file_part,
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
        },
    }
    resp = session.post(
        endpoint,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=GEMINI_TIMEOUT,
    )
    if resp.status_code != 200:
        error_message = resp.text[:300]
        error_status = ""
        try:
            data = resp.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and "error" in data:
            error = data.get("error", {}) or {}
            error_message = error.get("message", error_message)
            error_status = error.get("status", "")
        if KEY_PROFILE == "free" and _is_rate_limit_error(
            error_message, resp.status_code, error_status
        ):
            raise FreeTierLimitError(_format_free_tier_message(error_message))
        raise RuntimeError(f"HTTP {resp.status_code}: {error_message[:300]}")
    data = resp.json()
    if "error" in data:
        error = data.get("error", {}) or {}
        error_message = error.get("message", "Unknown API error")
        error_status = error.get("status", "")
        if KEY_PROFILE == "free" and _is_rate_limit_error(error_message, 200, error_status):
            raise FreeTierLimitError(_format_free_tier_message(error_message))
        raise RuntimeError(error_message)
    candidates = data.get("candidates", [])
    if not candidates:
        raise EmptyResponseError("Empty response candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    finish_reason = candidates[0].get("finishReason", "")
    usage = data.get("usageMetadata", {}) or {}
    return text, finish_reason, usage


def _call_gemini_with_retry(session, url, prompt, max_retries=GEMINI_MAX_RETRIES):
    attempt = 0
    timeout_retries = 0
    empty_retries = 0
    while True:
        try:
            text, finish_reason, usage = _call_gemini(session, url, prompt)
            return text, finish_reason, usage, attempt + 1, timeout_retries
        except FreeTierLimitError:
            raise
        except EmptyResponseError as e:
            attempt += 1
            empty_retries += 1
            if empty_retries >= EMPTY_RESPONSE_MAX_RETRIES:
                raise EmptyResponseError(
                    f"Empty response candidates (retries={empty_retries})",
                    attempts=attempt,
                    timeout_retries=timeout_retries,
                ) from e
            sleep_time = 5
            print(
                f"⚠️  API 空回應，{sleep_time}s 後重試 "
                f"({empty_retries}/{EMPTY_RESPONSE_MAX_RETRIES})"
            )
            time.sleep(sleep_time)
        except requests.exceptions.Timeout as e:
            attempt += 1
            timeout_retries += 1
            if not RETRY_FOREVER_ON_TIMEOUT and attempt >= max_retries:
                raise
            sleep_time = min(60, 2 ** attempt)
            print(f"⏱️  API 超時，{sleep_time}s 後重試: {e}")
            time.sleep(sleep_time)
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep_time = min(60, 2 ** attempt)
            print(f"⚠️  API 失敗，{sleep_time}s 後重試: {e}")
            time.sleep(sleep_time)


def _trim_overlap(prev_text, new_text, max_overlap=500):
    if not prev_text or not new_text:
        return new_text
    max_overlap = min(len(prev_text), len(new_text), max_overlap)
    for i in range(max_overlap, 0, -1):
        if prev_text[-i:] == new_text[:i]:
            return new_text[i:]
    return new_text


def get_transcript_text(url, session):
    """使用 Gemini API 從 YouTube URL 取得逐字稿。"""
    if not isinstance(url, str) or not url.strip().startswith("http"):
        return "[錯誤：無效的 YouTube 連結]", {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "continuations": 0,
            "retries": 0,
            "timeout_retries": 0,
            "finish_reason": "",
        }

    transcript_parts = []
    prompt = BASE_PROMPT
    last_tail = ""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    api_calls = 0
    continuations = 0
    retries = 0
    timeout_retries = 0
    finish_reason = ""

    for attempt in range(GEMINI_MAX_CONTINUATIONS + 1):
        try:
            text, finish_reason, usage, attempts, timeout_attempts = (
                _call_gemini_with_retry(session, url, prompt)
            )
        except EmptyResponseError as e:
            attempts = getattr(e, "attempts", 0)
            timeout_attempts = getattr(e, "timeout_retries", 0)
            api_calls += attempts
            retries += max(0, attempts)
            timeout_retries += timeout_attempts
            return f"[無法獲取逐字稿：API 空回應，已重試{attempts}次後跳過]", {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "api_calls": api_calls,
                "continuations": continuations,
                "retries": retries,
                "timeout_retries": timeout_retries,
                "finish_reason": finish_reason,
            }
        api_calls += 1
        retries += max(0, attempts - 1)
        timeout_retries += timeout_attempts
        input_tokens += _safe_int(usage.get("promptTokenCount"))
        output_tokens += _safe_int(usage.get("candidatesTokenCount"))
        total_tokens += _safe_int(usage.get("totalTokenCount"))
        if not text:
            return "[無法獲取逐字稿：空白回應]", {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "api_calls": api_calls,
                "continuations": continuations,
                "retries": retries,
                "timeout_retries": timeout_retries,
                "finish_reason": finish_reason,
            }

        if transcript_parts:
            text = _trim_overlap(transcript_parts[-1], text)
        transcript_parts.append(text)

        if finish_reason != "MAX_TOKENS":
            break

        continuations += 1
        lines = [line for line in text.splitlines() if line.strip()]
        last_tail = lines[-1] if lines else text[-200:]
        prompt = (
            "請從上一段逐字稿的結尾繼續輸出，不要重複之前內容。\n"
            f"上一段最後一句是：{last_tail}\n"
        )

    return "\n".join(transcript_parts), {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "api_calls": api_calls,
        "continuations": continuations,
        "retries": retries,
        "timeout_retries": timeout_retries,
        "finish_reason": finish_reason,
    }


def _is_checked(value):
    text = str(value).strip().lower()
    return text in {"[x]", "x", "✓", "✔", "✅", "true", "1", "yes"}


def _normalize_checkbox(value, checked):
    if checked:
        return "[x]"
    return "[ ]" if not _is_checked(value) else "[x]"


def _load_done_urls(output_dir):
    done = set()
    if not os.path.isdir(output_dir):
        return done
    for name in os.listdir(output_dir):
        if not name.endswith(".md"):
            continue
        path = os.path.join(output_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("* **影片連結**:"):
                        url = line.split(":", 1)[1].strip()
                        if url:
                            done.add(url)
        except Exception:
            continue
    return done


def _count_entries(path):
    try:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("## "):
                    count += 1
        return count
    except FileNotFoundError:
        return 0

def main():
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, _handle_sigint)

    start_time = time.monotonic()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_start = datetime.now().isoformat(timespec="seconds")
    csv_file = 'youtube_videos.csv'
    output_dir = 'transcript'
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    
    if not os.path.exists(csv_file):
        print(f"錯誤：找不到 {csv_file}，請確認檔案在同一資料夾內。")
        return
    if not GEMINI_API_KEY:
        print(
            "錯誤：請設定環境變數 GEMINI_API_KEY，"
            "或在同資料夾放置 .gemini_key_paid / .gemini_key_free。"
        )
        print("可用 GEMINI_KEY_PROFILE=paid|free 指定使用哪一把 key。")
        return
    os.makedirs(output_dir, exist_ok=True)

    print("正在讀取 CSV 檔案...")
    try:
        df = pd.read_csv(csv_file)
        # 清理欄位名稱
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"讀取 CSV 失敗: {e}")
        return

    # 確保序號與 checkbox 欄位存在
    if CHECK_COLUMN not in df.columns:
        df.insert(0, CHECK_COLUMN, "[ ]")
    if "序號" not in df.columns:
        checked_idx = df.columns.get_loc(CHECK_COLUMN)
        df.insert(checked_idx + 1, "序號", "")

    # 重新排序欄位：Checked 第一、序號第二
    cols = [CHECK_COLUMN, "序號"] + [
        c for c in df.columns if c not in ("序號", CHECK_COLUMN)
    ]
    df = df[cols]
    df["序號"] = range(1, len(df) + 1)

    # 檢查必要欄位
    if 'Title' not in df.columns or 'URL' not in df.columns:
        print("錯誤：CSV 必須包含 'Title' 和 'URL' 欄位。")
        print(f"目前的欄位有: {df.columns.tolist()}")
        return

    # 先根據已生成的檔案同步 checkbox
    done_urls = _load_done_urls(output_dir)
    for idx, row in df.iterrows():
        url = str(row.get("URL", "")).strip()
        checked = url in done_urls
        df.at[idx, CHECK_COLUMN] = _normalize_checkbox(row[CHECK_COLUMN], checked)
    df.to_csv(csv_file, index=False)

    max_new = None
    if MAX_VIDEOS:
        try:
            max_new = int(MAX_VIDEOS)
            if max_new <= 0:
                max_new = None
        except ValueError:
            print("警告：MAX_VIDEOS 必須是整數，已忽略。")

    # 設定批次大小
    batch_size = 10
    total_videos = len(df)
    
    print(f"總共發現 {total_videos} 支影片。")
    print(f"使用模型：{GEMINI_MODEL}")
    print("--------------------------------------------------")

    session = requests.Session()
    processed_new = 0
    success_count = 0
    fail_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    total_cost_known = True
    stop_reason = ""
    retry_index = None

    if RETRY_INDEX_RAW:
        try:
            retry_index = int(RETRY_INDEX_RAW)
            if retry_index <= 0:
                raise ValueError
        except ValueError:
            print("錯誤：RETRY_INDEX 必須是正整數。")
            return

    if retry_index is not None:
        if STOP_REQUESTED:
            stop_reason = "使用者中斷（Ctrl+C）"
        else:
            match = df[df["序號"] == retry_index]
            if match.empty:
                print(f"錯誤：找不到序號 {retry_index} 的影片。")
                return
            index = match.index[0]
            row = match.iloc[0]
            title = str(row["Title"]).strip()
            url = str(row["URL"]).strip()
            serial = row.get("序號", "")

            part_num = (retry_index - 1) // batch_size + 1
            part_start = (part_num - 1) * batch_size + 1
            part_end = min(part_num * batch_size, total_videos)
            filename = os.path.join(
                output_dir, f"易學基地頻道_影片逐字稿_Part_{part_num}.md"
            )
            header = _build_part_header(part_num, part_start, part_end)

            video_start = time.monotonic()
            print(f"單支補跑：序號 {retry_index} - {title[:30]}...")
            try:
                transcript_text, stats = get_transcript_text(url, session)
            except FreeTierLimitError as e:
                stop_reason = str(e)
                video_elapsed = int(time.monotonic() - video_start)
                _write_log_row(
                    log_path,
                    {
                        "工作階段ID": run_id,
                        "層級": "影片",
                        "開始時間": run_start,
                        "結束時間": datetime.now().isoformat(timespec="seconds"),
                        "耗時(秒)": video_elapsed,
                        "影片序號": serial,
                        "CSV行號": index + 1,
                        "影片標題": title,
                        "影片連結": url,
                        "寫入檔名": filename,
                        "完成狀態": "中斷",
                        "使用Key類型": KEY_PROFILE,
                        "中斷原因": stop_reason,
                        "錯誤訊息": stop_reason,
                        "輸入Token數": 0,
                        "輸出Token數": 0,
                        "總Token數": 0,
                        "預估成本(USD)": "",
                        "API呼叫次數": 0,
                        "續寫次數": 0,
                        "重試次數": 0,
                        "Timeout重試次數": 0,
                        "Finish原因": "",
                        "模型名稱": GEMINI_MODEL,
                        "媒體解析度": GEMINI_MEDIA_RESOLUTION,
                        "最大輸出Token": GEMINI_MAX_OUTPUT_TOKENS,
                        "超時秒數": GEMINI_TIMEOUT,
                        "每次間隔秒數": SLEEP_BETWEEN_CALLS,
                        "本次成功影片數": "",
                        "本次失敗影片數": "",
                        "本次總輸入Token": "",
                        "本次總輸出Token": "",
                        "本次總Token": "",
                        "本次總預估成本(USD)": "",
                    },
                )
                print(f"⛔️ {stop_reason}")
            else:
                if MERGE_LINES and not transcript_text.startswith("["):
                    transcript_text = _merge_transcript_lines(transcript_text)

                is_error = (
                    transcript_text.startswith("[無法獲取逐字稿")
                    or transcript_text.startswith("[錯誤")
                )
                input_tokens = stats.get("input_tokens", 0)
                output_tokens = stats.get("output_tokens", 0)
                video_total_tokens = stats.get("total_tokens", 0)
                api_calls = stats.get("api_calls", 0)
                continuations = stats.get("continuations", 0)
                retries = stats.get("retries", 0)
                timeout_retries = stats.get("timeout_retries", 0)
                finish_reason = stats.get("finish_reason", "")
                video_elapsed = int(time.monotonic() - video_start)
                cost = _estimate_cost(input_tokens, output_tokens)

                if not is_error:
                    entry = _build_entry(index, title, url, transcript_text)
                    action = _replace_entry_in_file(filename, header, url, entry)
                    df.at[index, CHECK_COLUMN] = "[x]"
                    df.to_csv(csv_file, index=False)
                    success_count += 1
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_tokens += video_total_tokens
                    if cost is None:
                        total_cost_known = False
                    else:
                        total_cost += cost
                    print(f"✅ 已更新 {filename} ({action})")
                else:
                    fail_count += 1
                    print("⚠️  單支補跑失敗，未改寫原本內容")

                _write_log_row(
                    log_path,
                    {
                        "工作階段ID": run_id,
                        "層級": "影片",
                        "開始時間": run_start,
                        "結束時間": datetime.now().isoformat(timespec="seconds"),
                        "耗時(秒)": video_elapsed,
                        "影片序號": serial,
                        "CSV行號": index + 1,
                        "影片標題": title,
                        "影片連結": url,
                        "寫入檔名": filename,
                        "完成狀態": "成功" if not is_error else "失敗",
                        "使用Key類型": KEY_PROFILE,
                        "中斷原因": "",
                        "錯誤訊息": "" if not is_error else transcript_text[:200],
                        "輸入Token數": input_tokens,
                        "輸出Token數": output_tokens,
                        "總Token數": video_total_tokens,
                        "預估成本(USD)": f"{cost:.6f}" if cost is not None else "",
                        "API呼叫次數": api_calls,
                        "續寫次數": continuations,
                        "重試次數": retries,
                        "Timeout重試次數": timeout_retries,
                        "Finish原因": finish_reason,
                        "模型名稱": GEMINI_MODEL,
                        "媒體解析度": GEMINI_MEDIA_RESOLUTION,
                        "最大輸出Token": GEMINI_MAX_OUTPUT_TOKENS,
                        "超時秒數": GEMINI_TIMEOUT,
                        "每次間隔秒數": SLEEP_BETWEEN_CALLS,
                        "本次成功影片數": "",
                        "本次失敗影片數": "",
                        "本次總輸入Token": "",
                        "本次總輸出Token": "",
                        "本次總Token": "",
                        "本次總預估成本(USD)": "",
                    },
                )
        if STOP_REQUESTED and not stop_reason:
            stop_reason = "使用者中斷（Ctrl+C）"
        elapsed = int(time.monotonic() - start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        if stop_reason:
            print(f"\n⛔️ 已中斷：{stop_reason}")
        else:
            print("\n單支補跑完成！")
        print(f"本次執行耗時：{hours}h {minutes}m {seconds}s")
        print(f"本次輸入Token：{total_input_tokens}")
        print(f"本次輸出Token：{total_output_tokens}")
        print(f"本次總Token：{total_tokens}")
        if total_cost_known:
            print(f"本次預估成本(USD)：{total_cost:.6f}")
        else:
            print("本次預估成本(USD)：(未設定價格或缺少 token 資訊)")

        _write_log_row(
            log_path,
            {
                "工作階段ID": run_id,
                "層級": "工作階段",
                "開始時間": run_start,
                "結束時間": datetime.now().isoformat(timespec="seconds"),
                "耗時(秒)": elapsed,
                "影片序號": "",
                "CSV行號": "",
                "影片標題": "",
                "影片連結": "",
                "寫入檔名": "",
                "完成狀態": "中斷" if stop_reason else "完成",
                "使用Key類型": KEY_PROFILE,
                "中斷原因": stop_reason,
                "錯誤訊息": stop_reason,
                "輸入Token數": "",
                "輸出Token數": "",
                "總Token數": "",
                "預估成本(USD)": "",
                "API呼叫次數": "",
                "續寫次數": "",
                "重試次數": "",
                "Timeout重試次數": "",
                "Finish原因": "",
                "模型名稱": GEMINI_MODEL,
                "媒體解析度": GEMINI_MEDIA_RESOLUTION,
                "最大輸出Token": GEMINI_MAX_OUTPUT_TOKENS,
                "超時秒數": GEMINI_TIMEOUT,
                "每次間隔秒數": SLEEP_BETWEEN_CALLS,
                "本次成功影片數": success_count,
                "本次失敗影片數": fail_count,
                "本次總輸入Token": total_input_tokens,
                "本次總輸出Token": total_output_tokens,
                "本次總Token": total_tokens,
                "本次總預估成本(USD)": f"{total_cost:.6f}" if total_cost_known else "",
            },
        )
        return

    # 分批處理
    for i in range(0, total_videos, batch_size):
        if STOP_REQUESTED and not stop_reason:
            stop_reason = "使用者中斷（Ctrl+C）"
            break
        batch_num = (i // batch_size) + 1

        batch_df = df.iloc[i : i + batch_size]
        # 建立檔名
        filename = os.path.join(output_dir, f"易學基地頻道_影片逐字稿_Part_{batch_num}.md")
        if os.path.exists(filename) and not OVERWRITE:
            existing_count = _count_entries(filename)
            if existing_count >= len(batch_df):
                print(f"略過 {filename} (已存在且完整，設 OVERWRITE=1 可覆寫)")
                continue
        else:
            existing_count = 0

        print(f"正在製作 {filename} (進度: {min(i+batch_size, total_videos)}/{total_videos})...")

        file_mode = "w"
        if os.path.exists(filename) and not OVERWRITE:
            file_mode = "a"
        with open(filename, file_mode, encoding="utf-8") as f:
            if file_mode == "w":
                f.write(f"# 易學基地頻道 影片逐字稿 Part {batch_num}\n")
                f.write(f"**範圍：影片 {i+1:03d} - {min(i+batch_size, total_videos):03d}**\n\n---\n\n")

            for local_idx, (index, row) in enumerate(batch_df.iterrows()):
                if STOP_REQUESTED and not stop_reason:
                    stop_reason = "使用者中斷（Ctrl+C）"
                    break
                if local_idx < existing_count:
                    continue
                if _is_checked(row.get(CHECK_COLUMN, "")):
                    continue
                if max_new is not None and processed_new >= max_new:
                    break

                title = str(row["Title"]).strip()
                url = str(row["URL"]).strip()
                serial = row.get("序號", "")
                video_start = time.monotonic()
                print(f"  - 處理中: {title[:30]}...")

                try:
                    transcript_text, stats = get_transcript_text(url, session)
                except FreeTierLimitError as e:
                    stop_reason = str(e)
                    video_elapsed = int(time.monotonic() - video_start)
                    _write_log_row(
                        log_path,
                        {
                            "工作階段ID": run_id,
                            "層級": "影片",
                            "開始時間": run_start,
                            "結束時間": datetime.now().isoformat(timespec="seconds"),
                            "耗時(秒)": video_elapsed,
                            "影片序號": serial,
                            "CSV行號": index + 1,
                            "影片標題": title,
                            "影片連結": url,
                            "寫入檔名": filename,
                            "完成狀態": "中斷",
                            "使用Key類型": KEY_PROFILE,
                            "中斷原因": stop_reason,
                            "錯誤訊息": stop_reason,
                            "輸入Token數": 0,
                            "輸出Token數": 0,
                            "總Token數": 0,
                            "預估成本(USD)": "",
                            "API呼叫次數": 0,
                            "續寫次數": 0,
                            "重試次數": 0,
                            "Timeout重試次數": 0,
                            "Finish原因": "",
                            "模型名稱": GEMINI_MODEL,
                            "媒體解析度": GEMINI_MEDIA_RESOLUTION,
                            "最大輸出Token": GEMINI_MAX_OUTPUT_TOKENS,
                            "超時秒數": GEMINI_TIMEOUT,
                            "每次間隔秒數": SLEEP_BETWEEN_CALLS,
                            "本次成功影片數": "",
                            "本次失敗影片數": "",
                            "本次總輸入Token": "",
                            "本次總輸出Token": "",
                            "本次總Token": "",
                            "本次總預估成本(USD)": "",
                        },
                    )
                    print(f"⛔️ {stop_reason}")
                    break

                f.write(f"## {index + 1}. {title}\n")
                f.write(f"* **影片連結**: {url}\n")
                f.write("* **逐字稿**:\n")

                if MERGE_LINES and not transcript_text.startswith("["):
                    transcript_text = _merge_transcript_lines(transcript_text)

                formatted_text = "\n".join(
                    [f"    > {line}" for line in transcript_text.splitlines()]
                )
                f.write(formatted_text)
                f.write("\n\n---\n\n")
                f.flush()

                is_error = (
                    transcript_text.startswith("[無法獲取逐字稿")
                    or transcript_text.startswith("[錯誤")
                )
                input_tokens = stats.get("input_tokens", 0)
                output_tokens = stats.get("output_tokens", 0)
                video_total_tokens = stats.get("total_tokens", 0)
                api_calls = stats.get("api_calls", 0)
                continuations = stats.get("continuations", 0)
                retries = stats.get("retries", 0)
                timeout_retries = stats.get("timeout_retries", 0)
                finish_reason = stats.get("finish_reason", "")
                video_elapsed = int(time.monotonic() - video_start)
                cost = _estimate_cost(input_tokens, output_tokens)

                if not is_error:
                    df.at[index, CHECK_COLUMN] = "[x]"
                    df.to_csv(csv_file, index=False)
                    processed_new += 1
                    success_count += 1
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_tokens += video_total_tokens
                    if cost is None:
                        total_cost_known = False
                    else:
                        total_cost += cost
                else:
                    fail_count += 1
                    print("⚠️  本支未成功，稍後可重跑補上")

                _write_log_row(
                    log_path,
                    {
                        "工作階段ID": run_id,
                        "層級": "影片",
                        "開始時間": run_start,
                        "結束時間": datetime.now().isoformat(timespec="seconds"),
                        "耗時(秒)": video_elapsed,
                        "影片序號": serial,
                        "CSV行號": index + 1,
                        "影片標題": title,
                        "影片連結": url,
                        "寫入檔名": filename,
                        "完成狀態": "成功" if not is_error else "失敗",
                        "使用Key類型": KEY_PROFILE,
                        "中斷原因": "",
                        "錯誤訊息": "" if not is_error else transcript_text[:200],
                        "輸入Token數": input_tokens,
                        "輸出Token數": output_tokens,
                        "總Token數": video_total_tokens,
                        "預估成本(USD)": f"{cost:.6f}" if cost is not None else "",
                        "API呼叫次數": api_calls,
                        "續寫次數": continuations,
                        "重試次數": retries,
                        "Timeout重試次數": timeout_retries,
                        "Finish原因": finish_reason,
                        "模型名稱": GEMINI_MODEL,
                        "媒體解析度": GEMINI_MEDIA_RESOLUTION,
                        "最大輸出Token": GEMINI_MAX_OUTPUT_TOKENS,
                        "超時秒數": GEMINI_TIMEOUT,
                        "每次間隔秒數": SLEEP_BETWEEN_CALLS,
                        "本次成功影片數": "",
                        "本次失敗影片數": "",
                        "本次總輸入Token": "",
                        "本次總輸出Token": "",
                        "本次總Token": "",
                        "本次總預估成本(USD)": "",
                    },
                )
                time.sleep(SLEEP_BETWEEN_CALLS)

            if stop_reason:
                break
            if max_new is not None and processed_new >= max_new:
                print("已達本次處理上限。")
                break

        if stop_reason:
            break
        print(f"\n✅ {filename} 完成！")

        if max_new is not None and processed_new >= max_new:
            break

    elapsed = int(time.monotonic() - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    if stop_reason:
        print(f"\n⛔️ 已中斷：{stop_reason}")
    else:
        print("\n全部任務完成！所有文件已生成。")
    print(f"本次執行耗時：{hours}h {minutes}m {seconds}s")
    print(f"本次輸入Token：{total_input_tokens}")
    print(f"本次輸出Token：{total_output_tokens}")
    print(f"本次總Token：{total_tokens}")
    if total_cost_known:
        print(f"本次預估成本(USD)：{total_cost:.6f}")
    else:
        print("本次預估成本(USD)：(未設定價格或缺少 token 資訊)")

    _write_log_row(
        log_path,
        {
            "工作階段ID": run_id,
            "層級": "工作階段",
            "開始時間": run_start,
            "結束時間": datetime.now().isoformat(timespec="seconds"),
            "耗時(秒)": elapsed,
            "影片序號": "",
            "CSV行號": "",
            "影片標題": "",
            "影片連結": "",
            "寫入檔名": "",
            "完成狀態": "中斷" if stop_reason else "完成",
            "使用Key類型": KEY_PROFILE,
            "中斷原因": stop_reason,
            "錯誤訊息": stop_reason,
            "輸入Token數": "",
            "輸出Token數": "",
            "總Token數": "",
            "預估成本(USD)": "",
            "API呼叫次數": "",
            "續寫次數": "",
            "重試次數": "",
            "Timeout重試次數": "",
            "Finish原因": "",
            "模型名稱": GEMINI_MODEL,
            "媒體解析度": GEMINI_MEDIA_RESOLUTION,
            "最大輸出Token": GEMINI_MAX_OUTPUT_TOKENS,
            "超時秒數": GEMINI_TIMEOUT,
            "每次間隔秒數": SLEEP_BETWEEN_CALLS,
            "本次成功影片數": success_count,
            "本次失敗影片數": fail_count,
            "本次總輸入Token": total_input_tokens,
            "本次總輸出Token": total_output_tokens,
            "本次總Token": total_tokens,
            "本次總預估成本(USD)": f"{total_cost:.6f}" if total_cost_known else "",
        },
    )

if __name__ == "__main__":
    main()
