from pathlib import Path

def test_research_repository_contains_required_components():
    root=Path(__file__).resolve().parents[1]
    required=["README.md","docs/experiment_protocol.md","docs/data_card.md",
              "paper/main.tex","paper/refs.bib","scripts/run_full_study.py",
              "src/pm25forecast/preprocessing.py","src/pm25forecast/pipeline.py"]
    assert all((root/p).exists() for p in required)

def test_paper_contains_a_valid_results_protocol():
    root=Path(__file__).resolve().parents[1]
    text=(root/"paper/main.tex").read_text(encoding="utf-8")
    dynamic_results="generated_results.tex" in text
    integrated_results=("\\section{Results}" in text and
                        "Locked-test forecast accuracy" in text)
    assert dynamic_results or integrated_results
    assert "XX.XX" not in text
    assert "TBD after execution" not in text
