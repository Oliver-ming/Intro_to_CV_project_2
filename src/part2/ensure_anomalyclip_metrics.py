"""Make AnomalyCLIP's test script write raw metrics next to its printed table.

The upstream script only logs percentages rounded to one decimal. This inserts
one marker and a JSON dump. It is idempotent.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "PROJECT2_METRICS_JSON"

_ANCHOR = '    logger.info("\\n%s", results)\n'

_INSERT = '''    logger.info("\\n%s", results)
    # PROJECT2_METRICS_JSON: raw scores, not the one-decimal table.
    import json
    metric_payload = {
        "dataset": args.dataset,
        "metrics": args.metrics,
        "per_category": saved_rows,
        "mean_image_auroc": float(np.mean(image_auroc_list)) if image_auroc_list else None,
        "mean_image_ap": float(np.mean(image_ap_list)) if image_ap_list else None,
        "mean_pixel_auroc": float(np.mean(pixel_auroc_list)) if pixel_auroc_list else None,
        "mean_pixel_aupro": float(np.mean(pixel_aupro_list)) if pixel_aupro_list else None,
    }
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metric_payload, handle, indent=2)
'''

_ROW_ANCHOR = "    table_ls = []\n"
_ROW_INSERT = '''    table_ls = []
    saved_rows = []
'''

_SAVE_ROW = "        table_ls.append(table)\n"
_SAVE_ROW_INSERT = '''        saved = {"category": obj}
        if args.metrics in ("image-level", "image-pixel-level"):
            saved["image_auroc"] = float(image_auroc)
            saved["image_ap"] = float(image_ap)
        if args.metrics in ("pixel-level", "image-pixel-level"):
            saved["pixel_auroc"] = float(pixel_auroc)
            saved["pixel_aupro"] = float(pixel_aupro)
        saved_rows.append(saved)
        table_ls.append(table)
'''


def ensure_anomalyclip_metrics(test_py: Path) -> bool:
    """Return True when the file was changed."""
    text = test_py.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    if _ANCHOR not in text or _ROW_ANCHOR not in text or _SAVE_ROW not in text:
        raise RuntimeError(f"{test_py} does not match the expected AnomalyCLIP test.py")
    text = text.replace(_ROW_ANCHOR, _ROW_INSERT, 1)
    text = text.replace(_SAVE_ROW, _SAVE_ROW_INSERT, 1)
    text = text.replace(_ANCHOR, _INSERT, 1)
    test_py.write_text(text, encoding="utf-8")
    return True
