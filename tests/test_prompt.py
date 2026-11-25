"""Prompt framing and the image grammar (downloads the small tokenizer files from the upstream repo)."""
import pytest

hf = pytest.importorskip("huggingface_hub")


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    for f in ("tokenizer.json", "chat_template.jinja", "unitron_extension_manifest.json"):
        try:
            hf.hf_hub_download("nvidia/Nemotron-Labs-Image-30B-A3B", f, local_dir=d)
        except Exception as e:  # no network
            pytest.skip(f"cannot download {f}: {e}")
    from nemotron_image_mlx.prompt import PromptTokenizer
    return PromptTokenizer(d)


def test_prompt_ends_with_empty_think(tok):
    text = tok.render("a fox", 1024, 1024)
    assert text.endswith("<|im_start|>assistant\n<think></think>")
    assert "Generate a 1024x1024 image.\n\na fox" in text


def test_schedule_round_trip(tok):
    from nemotron_image_mlx.pipeline import image_schedule
    w, h = 64, 48
    sched = image_schedule(tok, w, h)
    codes = iter(range(1000))
    ids = [1010, tok.ids.img_start] + [s if s is not None else tok.ids.vis_start + next(codes) for s in sched]
    out, grid, header = tok.parse_image(ids, w, h)
    assert grid == (3, 4) and header.strip() == "height=48 width=64" and out == list(range(12))


def test_bad_size_rejected(tok):
    with pytest.raises(ValueError):
        tok.encode("x", 100, 100)
