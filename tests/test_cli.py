"""CLI contract: exit codes, output formats, file artefacts."""

from __future__ import annotations

import csv
import json

import pytest

from imgsim.cli import main


def test_text_output_and_artifacts(corpus, tmp_path, capsys):
    out = tmp_path / "out"
    code = main([str(corpus["dir"]), "-o", str(out)])
    captured = capsys.readouterr()

    assert code == 0
    assert "group(s)" in captured.out
    assert (out / "report.html").exists()
    assert (out / "groups.csv").exists()
    assert (out / "pairs.csv").exists()
    assert (out / "features.npz").exists()


def test_json_output_is_valid(corpus, tmp_path, capsys):
    main([str(corpus["dir"]), "-o", str(tmp_path / "out"), "-f", "json", "-q"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "hash"
    assert payload["images"] >= 14
    assert isinstance(payload["groups"], list)
    assert payload["groups"] == sorted(payload["groups"], key=lambda g: -g["size"])


def test_csv_output_is_parseable(corpus, tmp_path, capsys):
    main([str(corpus["dir"]), "-o", str(tmp_path / "out"), "-f", "csv", "-q"])
    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
    assert rows
    assert set(rows[0]) == {"group", "size", "distance", "path"}


def test_no_report_skips_html(corpus, tmp_path):
    out = tmp_path / "out"
    main([str(corpus["dir"]), "-o", str(out), "--no-report", "-q"])
    assert not (out / "report.html").exists()


def test_fail_on_duplicates_exit_code(corpus, tmp_path):
    code = main([str(corpus["dir"]), "-o", str(tmp_path / "out"),
                 "--fail-on-duplicates", "-q"])
    assert code == 1


def test_fail_on_duplicates_passes_on_clean_corpus(corpus, tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "a.jpg").write_bytes((corpus["dir"] / "a0.jpg").read_bytes())
    code = main([str(clean), "-o", str(tmp_path / "out"), "--fail-on-duplicates", "-q"])
    assert code == 0


def test_reuse_cache_matches_fresh_run(corpus, tmp_path, capsys):
    out = tmp_path / "out"
    main([str(corpus["dir"]), "-o", str(out), "-f", "json", "-q"])
    fresh = json.loads(capsys.readouterr().out)

    main([str(corpus["dir"]), "-o", str(out), "-f", "json", "-q", "--reuse"])
    cached = json.loads(capsys.readouterr().out)

    assert fresh["groups"] == cached["groups"]


def test_stale_cache_falls_back_to_reindexing(corpus, tmp_path, capsys):
    out = tmp_path / "out"
    main([str(corpus["dir"]), "-o", str(out), "-q"])
    (out / "features.npz").write_bytes(b"corrupted")

    code = main([str(corpus["dir"]), "-o", str(out), "--reuse", "-q"])
    assert code == 0
    assert "re-indexing" in capsys.readouterr().err


def test_query_mode(corpus, tmp_path, capsys):
    code = main([
        str(corpus["dir"]), "-o", str(tmp_path / "out"), "-q",
        "--query", str(corpus["group_a"][0]), "-k", "3",
    ])
    assert code == 0
    assert "nearest to" in capsys.readouterr().out


def test_topk_file(corpus, tmp_path):
    out = tmp_path / "out"
    main([str(corpus["dir"]), "-o", str(out), "--topk", "2", "-q"])
    path = out / "topk.csv"
    assert path.exists()
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert rows
    assert set(rows[0]) == {"image", "rank", "neighbour", "distance"}


def test_missing_directory_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "ghost")])
    assert exc.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main([".", "--version"])
    assert exc.value.code == 0
    assert "imgsim" in capsys.readouterr().out
