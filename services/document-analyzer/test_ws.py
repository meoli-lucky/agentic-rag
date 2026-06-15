"""
Test WebSocket /ws/layout-analysis

Cách dùng:
    python test_ws.py <đường_dẫn_file_pdf_hoặc_ảnh>

Ví dụ:
    python test_ws.py /path/to/document.pdf
    python test_ws.py /path/to/scan.jpg
"""

import asyncio
import base64
import json
import sys
import time

try:
    import websockets
except ImportError:
    print("Cài đặt websockets trước: pip install websockets")
    sys.exit(1)


HOST = "localhost"
PORT = 8500
WS_URL = f"ws://{HOST}:{PORT}/ws/layout-analysis"


async def test(file_path: str):
    print(f"📄 File: {file_path}")
    print(f"🔗 Connecting to {WS_URL} ...\n")

    with open(file_path, "rb") as f:
        file_b64 = base64.b64encode(f.read()).decode()

    params = {
        "file_b64":           file_b64,
        "x_user_id":          "test-user",
        "x_conversation_id":  "test-conv",
        "x_document_id":      "test-doc-001",
        "storage_type":       "local",      # dùng local cho dễ test
        "confidence_threshold": 0.25,
        "sort":               "coordinates",
        "show_height":        False,
        "show_width":         False,
        "remove_page_header": False,
        "merge_suspicion":    True,
        "check_digital_text": True,
        "doc_recognizer":     True,
        "table_recognizer":   True,
        "smart_ocr":          True,
    }

    t_start = time.time()

    async with websockets.connect(WS_URL, max_size=200 * 1024 * 1024) as ws:
        print("✅ Connected!\n")
        await ws.send(json.dumps(params))
        print("📤 Params sent. Waiting for progress...\n")
        print("─" * 60)

        async for message in ws:
            frame = json.loads(message)
            event = frame["event"]

            if event == "progress":
                step      = frame.get("step", "?")
                step_name = frame.get("step_name", "")
                page      = frame.get("page")
                total_p   = frame.get("total_pages")
                msg       = frame.get("message", "")
                data      = frame.get("data")

                page_info = f"[Page {page}/{total_p}]" if page else ""
                print(f"  [Step {step}] {step_name} {page_info}")
                print(f"           → {msg}")
                if data:
                    for k, v in data.items():
                        print(f"             {k}: {v}")
                print()

            elif event == "complete":
                elapsed = time.time() - t_start
                result  = frame["result"]
                print("─" * 60)
                print(f"✅ COMPLETE in {elapsed:.1f}s")
                print(f"   Total elements : {len(result.get('data', []))}")
                print(f"   Total crops    : {result.get('total_crops', 0)}")
                print(f"   Result file    : {result.get('result_file_url', 'N/A')}")
                break

            elif event == "error":
                print(f"\n❌ ERROR: {frame['message']}")
                break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ws.py <file_path>")
        sys.exit(1)
    asyncio.run(test(sys.argv[1]))
