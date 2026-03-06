#!/usr/bin/env bash
set -euo pipefail

DIR="/Users/ryaotuix/rizz/reply_helper/backend/classifier/data/full_shot"
REPEAT="${1:-1}"
CONC="${2:-1}"
URL="http://localhost:8000/analyze-image"
DEVICE_ID="bench"

outdir="./bench_out_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$outdir"

files_txt="$outdir/files.txt"
find "$DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort > "$files_txt"
if [[ ! -s "$files_txt" ]]; then
  echo "No images found in: $DIR"
  exit 1
fi

num_files="$(wc -l < "$files_txt" | tr -d ' ')"
echo "Found $num_files images in: $DIR"
echo "REPEAT=$REPEAT, CONCURRENCY=$CONC"
echo "OUTDIR=$outdir"
echo

csv="$outdir/results.csv"
echo "run_idx,file,success,num_detections,yolo_detect_ms,ocr_ms,llm_ms,total_ms,fallback_count,http_code,result_json_path,turns_text_path,meta_path" > "$csv"

run_one() {
  local run_idx="$1"
  local f="$2"

  local base stem json_path txt_path meta_path
  base="$(basename "$f")"
  stem="${base%.*}"

  json_path="$outdir/${run_idx}__${stem}.json"
  txt_path="$outdir/${run_idx}__${stem}.turns.txt"
  meta_path="$outdir/${run_idx}__${stem}.meta.txt"

  # ✅ body는 파일로 직접 저장 → stdout 섞임 원천 차단
  # ✅ stderr는 meta로 저장 (경고/에러 추적 가능)
  http_code="$(curl --no-progress-meter -sS -o "$json_path" -w "%{http_code}" \
    -X POST "$URL" \
    -F "device_id=$DEVICE_ID" \
    -F "image=@$f" 2>"$meta_path" || true)"

  {
    echo "http_code=$http_code"
    echo "file=$f"
    echo "json_path=$json_path"
  } >> "$meta_path"

  python3 - "$run_idx" "$f" "$json_path" "$txt_path" "$meta_path" "$http_code" >> "$csv" <<'PY'
import sys, json

run_idx, file_path, json_path, txt_path, meta_path, http_code = sys.argv[1:7]

def fail(msg):
    with open(txt_path, "w", encoding="utf-8") as w:
        w.write(msg + "\n")
        w.write(f"http_code={http_code}\n\n")
        w.write("(meta_head)\n")
        try:
            w.write(open(meta_path, "r", encoding="utf-8", errors="ignore").read()[:2000] + "\n")
        except Exception:
            pass
        w.write("\n(body_head)\n")
        try:
            raw_s = open(json_path, "r", encoding="utf-8", errors="ignore").read()
            w.write(raw_s[:2000])
        except Exception as e:
            w.write(f"(read_failed) {e}\n")
    print(",".join([run_idx, json.dumps(file_path), "false","","","","","","","",http_code,json.dumps(json_path),json.dumps(txt_path),json.dumps(meta_path)]))
    sys.exit(0)

try:
    raw_s = open(json_path, "r", encoding="utf-8", errors="ignore").read()
except Exception as e:
    fail(f"(read_failed) {e}")

raw_s_strip = raw_s.strip()
if not raw_s_strip:
    fail("(empty_body) response body was empty")

# robust slice: outermost JSON object
l = raw_s_strip.find("{")
r = raw_s_strip.rfind("}")
if l != -1 and r != -1 and r > l:
    raw_json = raw_s_strip[l:r+1]
else:
    raw_json = raw_s_strip

try:
    resp = json.loads(raw_json)
except Exception:
    fail("(parse_failed) response was not valid JSON")

if not isinstance(resp, dict):
    fail("(parse_failed) JSON was not an object")

t = resp.get("timings") or {}
turns = resp.get("turns") or []
debug_lines = resp.get("debug_lines") or []
fallback_count = sum(1 for line in debug_lines if "fallback:easyocr" in str(line))

with open(txt_path, "w", encoding="utf-8") as w:
    for i, tr in enumerate(turns, 1):
        who = tr.get("who", "")
        text = tr.get("text", "")
        w.write(f"{i}. [{who}] {text}\n")

row = [
    run_idx,
    json.dumps(file_path),
    str(bool(resp.get("success", False))).lower(),
    str(resp.get("num_detections", "")),
    str(t.get("yolo_detect_ms", "")),
    str(t.get("ocr_ms", "")),
    str(t.get("llm_ms", "")),
    str(t.get("total_ms", "")),
    str(fallback_count),
    http_code,
    json.dumps(json_path),
    json.dumps(txt_path),
    json.dumps(meta_path),
]
print(",".join(row))
PY
}

export -f run_one
export URL DEVICE_ID outdir csv

jobs="$outdir/jobs.txt"
: > "$jobs"
for ((i=1;i<=REPEAT;i++)); do
  while IFS= read -r f; do
    echo "${i}|${f}" >> "$jobs"
  done < "$files_txt"
done

total_jobs="$(wc -l < "$jobs" | tr -d ' ')"
echo "Total requests: $total_jobs"
echo

cat "$jobs" | xargs -n 1 -P "$CONC" -I {} bash -lc '
  IFS="|" read -r run_idx f <<< "{}"
  run_one "$run_idx" "$f"
'

echo
echo "Saved: $csv"
echo "Per-image outputs in: $outdir"
echo

python3 - <<PY
import pandas as pd
df = pd.read_csv("$csv")
for c in ["num_detections","yolo_detect_ms","ocr_ms","llm_ms","total_ms","fallback_count"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def q(s,p): return float(s.quantile(p))

print("rows=", len(df), "success_rate=", round((df["success"]==True).mean(), 3))
print("avg_detections=", round(df["num_detections"].mean(), 2))
print("avg_fallbacks=", round(df["fallback_count"].mean(), 2))

for col in ["yolo_detect_ms","ocr_ms","total_ms"]:
    s=df[col].dropna()
    if len(s)==0: 
        continue
    print(f"\n[{col}] mean={s.mean():.1f} p50={q(s,0.50):.1f} p95={q(s,0.95):.1f} max={s.max():.1f}")

slow = df.sort_values("total_ms", ascending=False).head(5)[["file","num_detections","fallback_count","ocr_ms","total_ms","turns_text_path","http_code"]]
print("\n[top5_slowest]")
print(slow.to_string(index=False))
PY
