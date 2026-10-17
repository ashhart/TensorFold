"""edit_plan.py and the two edit scripts' use of it: noise seed, sizes, and a several-picture ComfyUI workflow."""
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spark"))
import edit_plan as P  # noqa: E402


def files(*contents):
    folder = Path(tempfile.mkdtemp())
    out = []
    for index, content in enumerate(contents):
        (folder / f"{index}.png").write_bytes(content)
        out.append(str(folder / f"{index}.png"))
    return out


def test_the_noise_belongs_to_the_seed_and_the_pictures():
    a, b = files(b"picture a", b"picture b")
    # the same seed and picture: the same noise, so an edit can be made again
    assert P.noise_seed(11, [a]) == P.noise_seed(11, [a])
    # the production run's case: a picture made at seed 11, edited at seed 11, never starts from the same draw
    assert P.noise_seed(11, [a]) != P.noise_seed(11, [b])
    assert P.noise_seed(11, [a]) != 11 and P.noise_seed(11, [a]) != P.noise_seed(12, [a])
    # a further picture, or the same two the other way round, is another edit
    assert len({P.noise_seed(11, [a]), P.noise_seed(11, [a, b]), P.noise_seed(11, [b, a])}) == 3
    assert 0 <= P.noise_seed(2**40, [a]) < 2**48


def test_an_edit_comes_back_at_the_pictures_own_size():
    # the run's keyframes: 1280x736 came back 1344x768 under the model's own rule
    assert P.canvas(1280, 736, "model", 1024)["size"] == (1344, 768)
    own = P.canvas(1280, 736)
    assert own["made"] == own["size"] == own["source"] == (1280, 736) and not own["note"]
    # a side off the model's grid: made on the grid, returned at the picture's own size
    off = P.canvas(1280, 720)
    assert off["size"] == (1280, 720) and off["made"][0] == 1280 and off["made"][1] % 32 == 0 and abs(off["made"][1] - 720) <= 16
    assert P.canvas(1024, 1024)["made"] == (1024, 1024)


def test_a_picture_outside_what_the_model_edits_at_says_so():
    big = P.canvas(4000, 3000)
    assert big["made"] == big["size"] and big["size"][0] * big["size"][1] <= 1536 * 1536 * 1.03 and "larger" in big["note"]
    assert abs(big["size"][0] / big["size"][1] - 4 / 3) < 0.03
    # a machine with less memory edits a large picture at fewer pixels
    capped = P.canvas(4000, 3000, most=1280 * 1280)
    assert capped["size"][0] * capped["size"][1] <= 1280 * 1280 * 1.03 and P.canvas(1280, 736, most=1280 * 1280)["size"] == (1280, 736)
    small = P.canvas(320, 240)
    assert small["size"][0] * small["size"][1] >= 512 * 512 * 0.95 and "smaller" in small["note"]


def test_a_size_asked_for_is_the_size_returned():
    same = P.canvas(1280, 736, asked=(1920, 1104))
    assert same["size"] == (1920, 1104) and same["made"] == same["source"] and same["made"][0] == 1920 and same["made"][1] in (1088, 1120)
    other = P.canvas(1280, 736, asked=(1024, 1024))
    # another shape: the picture keeps its own beside the square result
    assert other["made"] == (1024, 1024) and other["source"] != other["made"] and abs(other["source"][0] / other["source"][1] - 1280 / 736) < 0.05


def test_a_size_no_edit_is_made_at_is_refused():
    assert P.refused(1280, 736) == "" and P.refused(1536, 1536) == ""
    assert "between" in P.refused(100, 800) and "outside" in P.refused(4096, 4096) and "outside" in P.refused(300, 300)


def test_further_pictures_are_about_as_large_as_the_edit():
    wide, tall = P.beside(3000, 2000, (1280, 736))
    assert wide % 32 == 0 and tall % 32 == 0 and 0.85 < wide * tall / (1280 * 736) < 1.15 and abs(wide / tall - 1.5) < 0.05


def test_the_workflow_of_an_edit_of_several_pictures():
    import spark_edit as E

    args = types.SimpleNamespace(base=False, negative="", steps=40, cfg=1.0)
    w = E.workflow(args, ["a.png", "b.png", "c.png"], "the man of image 2 in the room of image 1", 123456, [])
    node = w["5"]["inputs"]
    assert [node[f"images.image_{i}"] for i in (1, 2, 3)] == [["20", 0], ["21", 0], ["22", 0]]
    assert [w[str(n)]["inputs"]["image"] for n in (20, 21, 22)] == ["a.png", "b.png", "c.png"]
    # each picture stays at the size it was brought to, and the edit is made on the first picture's
    assert node["resolution"] == 0 and w["12"]["inputs"]["latent_image"] == ["5", 2] and "6" not in w
    assert w["8"]["inputs"]["noise_seed"] == 123456
    # a size of another shape, asked for by hand
    w = E.workflow(args, ["a.png"], "wider", 1, [], (1024, 1024))
    assert w["6"]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1} and w["12"]["inputs"]["latent_image"] == ["6", 0]


def test_both_scripts_take_the_same_arguments():
    for script in ("qwen_image_edit.py", "spark/spark_edit.py"):
        text = (ROOT / script).read_text()
        assert 'action="append"' in text and "edit_plan.noise_seed(args.seed, args.image)" in text and '"--size"' in text
    for script in ("scripts/edit.sh", "spark/edit.sh"):
        text = (ROOT / script).read_text()
        assert '--size "${SIZE:-source}"' in text and "IMAGES" in text


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
