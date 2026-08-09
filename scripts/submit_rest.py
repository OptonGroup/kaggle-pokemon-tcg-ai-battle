#!/usr/bin/env python3
"""submit_rest.py — submit via the Kaggle SDK using a KGAT_ bearer token.

The `kaggle` CLI still authenticates through ~/.kaggle/kaggle.json (username +
key) and rejects the KGAT_ tokens the web UI now issues, so the CLI path is
unusable with an OAuth-era token. The SDK underneath does accept one — this
drives it directly.

    KAGGLE_API_TOKEN=KGAT_... python3 scripts/submit_rest.py <file.tar.gz> "<message>"
"""
from __future__ import annotations

import os
import sys

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiStartSubmissionUploadRequest,
)
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest


COMPETITION = "pokemon-tcg-ai-battle"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    path, message = argv[1], argv[2]
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("KAGGLE_API_TOKEN is not set", file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    size = os.path.getsize(path)
    print(f"[submit] {path} ({size/1e6:.1f} MB) -> {COMPETITION}")
    print(f"[submit] message: {message}")

    client = KaggleClient(api_token=token)

    req = ApiStartSubmissionUploadRequest()
    req.competition_name = COMPETITION
    req.file_name = os.path.basename(path)
    req.content_length = size
    req.last_modified_epoch_seconds = int(os.path.getmtime(path))

    started = client.competitions.competition_api_client.start_submission_upload(req)
    token_str = getattr(started, "token", None)
    upload_url = getattr(started, "create_url", None)
    print(f"[submit] upload slot: token={'ok' if token_str else 'MISSING'}")

    if not upload_url or not token_str:
        print(f"[submit] unexpected response: {started}", file=sys.stderr)
        return 1

    with open(path, "rb") as fh:
        import requests

        resp = requests.put(upload_url, data=fh, timeout=600)
    print(f"[submit] upload: HTTP {resp.status_code}")
    if resp.status_code not in (200, 201):
        print(resp.text[:400], file=sys.stderr)
        return 1

    create = ApiCreateSubmissionRequest()
    create.competition_name = COMPETITION
    create.blob_file_tokens = token_str
    create.submission_description = message
    result = client.competitions.competition_api_client.create_submission(create)
    print(f"[submit] RESULT: {getattr(result, 'message', result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
