"""CLI script for running latency analytics and bottleneck analysis.

Usage:
    python -m evaluation.analytics.cli --mock --limit 30
"""

import argparse
import csv
import json
from pathlib import Path

from evaluation.analytics.analyzer import LatencyAnalyzer
from evaluation.baseline import load_baseline
from evaluation.runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(description="HH Goa 2026 Latency Analytics & Bottleneck Identification")
    parser.add_argument("--mode", type=str, default="full", choices=["text_e2e", "voice_e2e", "full"], help="Run mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode")
    parser.add_argument("--output-dir", type=str, default="data/latency_analytics", help="Output directory")

    args = parser.parse_args()

    print(f"\n============================================================")
    print(f"  HH Goa 2026 — Phase 9.2 Latency Analytics & Bottlenecks")
    print(f"============================================================")

    # 1. Run evaluation suite to collect authoritative latency telemetry
    runner = EvaluationRunner(offline_mock=args.mock, seed=args.seed)
    cases = runner.loader.filter(limit=args.limit, seed=args.seed)

    raw_results = []
    harness = None
    voice_orch = None
    if not args.mock:
        from orchestration.harness.service import RAGHarness
        from voice.orchestrator import VoiceRAGOrchestrator
        harness = RAGHarness()
        voice_orch = VoiceRAGOrchestrator(rag_harness=harness)

    for case in cases:
        if args.mock:
            res = runner._run_mock_item(case, args.mode)
        else:
            res = runner._run_real_item(case, args.mode, harness, voice_orch)
        raw_results.append(res)

    # 2. Analyze Latency Telemetry
    baseline_report = load_baseline()
    dummy_meta = runner.run(mode=args.mode, limit=1).metadata
    analytics = LatencyAnalyzer.analyze_results(
        results=raw_results,
        report_meta=dummy_meta,
        baseline_report=baseline_report,
    )

    # 3. Export Machine-Readable Reports
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"latency_report_{analytics.metadata.run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analytics.model_dump(), f, indent=2, ensure_ascii=False)

    csv_path = out_dir / f"latency_report_{analytics.metadata.run_id}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Sample Count", analytics.sample_count])
        writer.writerow(["Text RAG P50 (ms)", analytics.text_latency.p50])
        writer.writerow(["Voice RAG P50 (ms)", analytics.voice_latency.p50])
        writer.writerow(["STT P50 (ms)", analytics.stt_latency.p50])
        writer.writerow(["Voice Overhead (ms)", analytics.voice_overhead_ms])
        writer.writerow(["Primary Bottleneck", f"{analytics.primary_bottleneck} ({analytics.primary_bottleneck_pct}%)"])
        writer.writerow(["Secondary Bottleneck", f"{analytics.secondary_bottleneck} ({analytics.secondary_bottleneck_pct}%)"])
        writer.writerow(["<200ms RAG Target", analytics.target_200ms_rag.status])
        writer.writerow(["<200ms Voice Target", analytics.target_200ms_voice.status])

    # 4. Display Clean Summary Report
    print("\n------------------------------------------------------------")
    print("AUTHORITATIVE LATENCY PERCENTILES")
    print("------------------------------------------------------------")
    print(f"Sample Count              : {analytics.sample_count}")
    print(f"Text RAG P50 (Median)     : {analytics.text_latency.p50:.3f} ms")
    print(f"Text RAG P70              : {analytics.text_latency.p70:.3f} ms")
    print(f"Text RAG P90              : {analytics.text_latency.p90:.3f} ms")
    print(f"Text RAG P95              : {analytics.text_latency.p95:.3f} ms")
    print(f"Text RAG P99              : {analytics.text_latency.p99:.3f} ms")
    print(f"Text RAG P100 (Max)       : {analytics.text_latency.p100:.3f} ms\n")

    print(f"Voice-to-Answer P50       : {analytics.voice_latency.p50:.3f} ms")
    print(f"Voice-to-Answer P90       : {analytics.voice_latency.p90:.3f} ms")
    print(f"Voice-to-Answer P95       : {analytics.voice_latency.p95:.3f} ms")
    print(f"Sarvam STT P50            : {analytics.stt_latency.p50:.3f} ms")
    print(f"Voice Latency Overhead   : {analytics.voice_overhead_ms:.3f} ms\n")

    print("------------------------------------------------------------")
    print("<200ms TARGET EVALUATION")
    print("------------------------------------------------------------")
    print(f"RAG Pipeline Target       : {analytics.target_200ms_rag.status} (P50: {analytics.target_200ms_rag.p50_ms:.1f}ms)")
    print(f"Voice-to-Answer Target    : {analytics.target_200ms_voice.status} (P50: {analytics.target_200ms_voice.p50_ms:.1f}ms)\n")

    print("------------------------------------------------------------")
    print("PIPELINE BOTTLENECK IDENTIFICATION")
    print("------------------------------------------------------------")
    print(f"Primary Bottleneck        : {analytics.primary_bottleneck} ({analytics.primary_bottleneck_pct:.1f}% contribution)")
    print(f"Secondary Bottleneck      : {analytics.secondary_bottleneck} ({analytics.secondary_bottleneck_pct:.1f}% contribution)\n")

    print(f"Exported analytics JSON to: {json_path}")
    print(f"Exported analytics CSV to : {csv_path}\n")


if __name__ == "__main__":
    main()
