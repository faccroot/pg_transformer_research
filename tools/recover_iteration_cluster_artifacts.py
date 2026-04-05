#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from tools.run_iteration_saved_diagnostics import build_run_entries, load_json
    from tools.update_iteration_observed_results import build_observed_results, collect_search_roots
except ImportError:
    from run_iteration_saved_diagnostics import build_run_entries, load_json
    from update_iteration_observed_results import build_observed_results, collect_search_roots


ROOT = Path(__file__).resolve().parents[1]
MINI_HOSTS = tuple(f"mini{i:02d}" for i in range(1, 15))
RECOVER_SUFFIXES = (
    "_mlx_model.npz",
    "_mlx_model.int8.ptz",
    "_int8zlib.pklz",
    "_trace_pretrain_model.npz",
    "_hardmax_controller_init.npz",
    ".summary.json",
    ".txt",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recover saved MLX artifacts/logs for a generated iteration from the Mac Mini job dirs."
    )
    p.add_argument("iteration_dir", help="Generated iteration directory containing manifest.json and configs/")
    p.add_argument(
        "--output-dir",
        default="",
        help="Local destination for recovered files. Defaults to <iteration_dir>/artifacts",
    )
    p.add_argument(
        "--host",
        action="append",
        default=[],
        help="Restrict search to specific Mini hosts. Can be passed multiple times.",
    )
    p.add_argument(
        "--suffix",
        action="append",
        default=[],
        help="Additional filename suffix to recover. Defaults include model artifacts and the main log txt.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-copy files that already exist locally.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned SSH/SCP actions without executing them.",
    )
    p.add_argument(
        "--connect-timeout",
        type=float,
        default=2.0,
        help="SSH connect timeout in seconds per host probe.",
    )
    p.add_argument(
        "--skip-observed-results",
        action="store_true",
        help="Do not refresh <iteration_dir>/observed_results.json after recovery.",
    )
    p.add_argument(
        "--run-hardmax-transfer-diagnostics",
        action="store_true",
        help="Run the hardmax transfer diagnostics wrapper after recovery.",
    )
    p.add_argument("--diagnostics-python", default=sys.executable, help="Python executable for the diagnostics wrapper.")
    p.add_argument("--diagnostics-output-dir", default="", help="Optional diagnostics output dir.")
    p.add_argument("--diagnostics-skip-existing", action="store_true", help="Pass --skip-existing to the diagnostics wrapper.")
    p.add_argument("--diagnostics-skip-residual", action="store_true", help="Pass --skip-residual to the diagnostics wrapper.")
    p.add_argument("--diagnostics-remote-analyzers", action="store_true", help="Run diagnostics controller/causal/factor analyzers via Mini dispatch.")
    p.add_argument("--diagnostics-remote-host", default="", help="Optional Mini host for remote analyzers.")
    p.add_argument("--diagnostics-remote-dispatch", default="", help="Optional dispatch.sh path for remote analyzers.")
    p.add_argument("--diagnostics-remote-repo-bundle", default="", help="Optional repo bundle for remote analyzers.")
    p.add_argument("--diagnostics-remote-keep-bundle", action="store_true", help="Keep auto-built repo bundle during remote diagnostics.")
    p.add_argument("--diagnostics-include-causal-ablation", action="store_true", help="Emit causal ablation JSONs in the post-recovery diagnostics run.")
    p.add_argument("--diagnostics-include-logit-factors", action="store_true", help="Emit logit factor JSONs in the post-recovery diagnostics run.")
    p.add_argument("--diagnostics-val-max-seqs", type=int, default=-1, help="Optional override for the post-recovery diagnostics wrapper.")
    p.add_argument("--diagnostics-eval-seq-len", type=int, default=-1, help="Optional override for the post-recovery diagnostics wrapper.")
    p.add_argument("--diagnostics-analysis-max-batches", type=int, default=-1, help="Optional override for the post-recovery diagnostics wrapper.")
    p.add_argument("--diagnostics-factor-max-batches", type=int, default=-1, help="Optional override for factor analysis max batches.")
    p.add_argument("--diagnostics-factor-num-factors", type=int, default=-1, help="Optional override for factor analysis num factors.")
    p.add_argument("--diagnostics-factor-top-tokens", type=int, default=-1, help="Optional override for factor analysis top token count.")
    return p.parse_args()


def ssh_find_remote_path(host: str, filename: str, *, timeout: float) -> str:
    quoted = shlex.quote(f"*{filename}")
    timeout_int = max(int(math.ceil(float(timeout))), 1)
    cmd = [
        "ssh",
        "-o",
        f"ConnectTimeout={timeout_int}",
        host,
        f"find ~/jobs -path {quoted} 2>/dev/null | tail -n 1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def scp_copy(remote_spec: str, dest_path: Path, *, dry_run: bool) -> None:
    cmd = ["scp", remote_spec, str(dest_path)]
    if dry_run:
        print("DRY-RUN", " ".join(cmd))
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def run_subprocess(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def hardmax_diagnostics_command(args: argparse.Namespace, iteration_dir: Path) -> list[str]:
    wrapper = ROOT / "tools" / "run_iteration_hardmax_transfer_diagnostics.py"
    cmd = [
        str(Path(args.diagnostics_python).expanduser()),
        str(wrapper),
        str(iteration_dir),
    ]
    if args.diagnostics_output_dir:
        cmd.extend(["--output-dir", str(Path(args.diagnostics_output_dir).expanduser().resolve())])
    if args.diagnostics_skip_existing:
        cmd.append("--skip-existing")
    if args.diagnostics_skip_residual:
        cmd.append("--skip-residual")
    if args.diagnostics_remote_analyzers:
        cmd.append("--remote-analyzers")
    if args.diagnostics_remote_host:
        cmd.extend(["--remote-host", str(args.diagnostics_remote_host)])
    if args.diagnostics_remote_dispatch:
        cmd.extend(["--remote-dispatch", str(args.diagnostics_remote_dispatch)])
    if args.diagnostics_remote_repo_bundle:
        cmd.extend(["--remote-repo-bundle", str(Path(args.diagnostics_remote_repo_bundle).expanduser().resolve())])
    if args.diagnostics_remote_keep_bundle:
        cmd.append("--remote-keep-bundle")
    if args.diagnostics_include_causal_ablation:
        cmd.append("--include-causal-ablation")
    if args.diagnostics_include_logit_factors:
        cmd.append("--include-logit-factors")
    if args.diagnostics_val_max_seqs > 0:
        cmd.extend(["--val-max-seqs", str(int(args.diagnostics_val_max_seqs))])
    if args.diagnostics_eval_seq_len > 0:
        cmd.extend(["--eval-seq-len", str(int(args.diagnostics_eval_seq_len))])
    if args.diagnostics_analysis_max_batches > 0:
        cmd.extend(["--analysis-max-batches", str(int(args.diagnostics_analysis_max_batches))])
    if args.diagnostics_factor_max_batches > 0:
        cmd.extend(["--factor-max-batches", str(int(args.diagnostics_factor_max_batches))])
    if args.diagnostics_factor_num_factors > 0:
        cmd.extend(["--factor-num-factors", str(int(args.diagnostics_factor_num_factors))])
    if args.diagnostics_factor_top_tokens > 0:
        cmd.extend(["--factor-top-tokens", str(int(args.diagnostics_factor_top_tokens))])
    return cmd


def main() -> None:
    args = parse_args()
    iteration_dir = Path(args.iteration_dir).expanduser().resolve()
    manifest = load_json(iteration_dir / "manifest.json")
    entries = build_run_entries(iteration_dir, manifest)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (iteration_dir / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    hosts = tuple(args.host) if args.host else MINI_HOSTS
    suffixes = tuple(dict.fromkeys((*RECOVER_SUFFIXES, *tuple(args.suffix))))

    recovered: dict[str, list[dict[str, str]]] = {}
    partial_missing: dict[str, list[str]] = {}
    unrecovered: dict[str, list[str]] = {}

    for entry in entries:
        run_records: list[dict[str, str]] = []
        missing_suffixes: list[str] = []
        for suffix in suffixes:
            filename = f"{entry.run_id}{suffix}"
            local_dest = output_dir / filename
            if args.skip_existing and local_dest.exists():
                run_records.append(
                    {
                        "suffix": suffix,
                        "host": "local",
                        "remote_path": "",
                        "local_path": str(local_dest),
                        "status": "exists",
                    }
                )
                continue
            found_host = ""
            found_remote = ""
            for host in hosts:
                remote_path = ssh_find_remote_path(host, filename, timeout=float(args.connect_timeout))
                if remote_path:
                    found_host = host
                    found_remote = remote_path
                    break
            if not found_remote:
                missing_suffixes.append(suffix)
                continue
            scp_copy(f"{found_host}:{found_remote}", local_dest, dry_run=args.dry_run)
            run_records.append(
                {
                    "suffix": suffix,
                    "host": found_host,
                    "remote_path": found_remote,
                    "local_path": str(local_dest),
                    "status": "copied" if not args.dry_run else "planned",
                }
            )
        if run_records:
            recovered[entry.run_slug] = run_records
            if missing_suffixes:
                partial_missing[entry.run_slug] = missing_suffixes
        elif missing_suffixes:
            unrecovered[entry.run_slug] = missing_suffixes

    summary = {
        "iteration_dir": str(iteration_dir),
        "output_dir": str(output_dir),
        "hosts": list(hosts),
        "suffixes": list(suffixes),
        "recovered": recovered,
        "partial_missing": partial_missing,
        "unrecovered": unrecovered,
    }
    summary_path = output_dir / "recovery_summary.json"
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.run_hardmax_transfer_diagnostics:
            run_subprocess(hardmax_diagnostics_command(args, iteration_dir), dry_run=True)
        return
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    observed_results_path = iteration_dir / "observed_results.json"
    if not args.skip_observed_results:
        observed_payload = build_observed_results(
            iteration_dir,
            check_remote=False,
            search_roots=collect_search_roots(iteration_dir, [str(output_dir)]),
        )
        observed_results_path.write_text(json.dumps(observed_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diagnostics_summary_path = ""
    if args.run_hardmax_transfer_diagnostics:
        diagnostics_cmd = hardmax_diagnostics_command(args, iteration_dir)
        run_subprocess(diagnostics_cmd, dry_run=args.dry_run)
        diagnostics_summary_path = str(
            (Path(args.diagnostics_output_dir).expanduser().resolve() if args.diagnostics_output_dir else (iteration_dir / "hardmax_transfer_diagnostics")) / "controller_summary.md"
        )
    print(
        json.dumps(
            {
                "runs_with_recovered_files": len(recovered),
                "fully_recovered_runs": sum(1 for slug in recovered if slug not in partial_missing),
                "partially_recovered_runs": len(partial_missing),
                "unrecovered_runs": len(unrecovered),
                "summary_path": str(summary_path),
                "observed_results_path": str(observed_results_path) if not args.skip_observed_results else "",
                "diagnostics_summary_path": diagnostics_summary_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
