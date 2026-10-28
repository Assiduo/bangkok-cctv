"""Fine-tuning helpers: label IO, box matching and the Ultralytics-style output decoder."""
import numpy as np

from cctv.detectors import FT_TO_COCO, FinetunedDetector
from cctv.labeling import dets_to_ft, iou, match, read_labels, write_labels


def test_label_roundtrip(tmp_path):
    f = tmp_path / "a.txt"
    write_labels(f, [(2, 0.9, 10, 20, 110, 80), (3, 0.8, 200, 100, 220, 140), (9, 0.9, 0, 0, 5, 5)], 352, 288)
    got = read_labels(f, 352, 288)
    assert [g[0] for g in got] == [0, 1]                       # car->0, motorcycle->1, traffic light dropped
    assert np.allclose(got[0][1:], (10, 20, 110, 80), atol=0.5)


def test_iou_and_match():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    gt = [(0, 0, 0, 10, 10), (0, 50, 50, 60, 60), (1, 100, 100, 110, 110)]
    pred = [(0, 0.9, 1, 1, 10, 10), (1, 0.8, 100, 100, 110, 110), (0, 0.7, 200, 200, 210, 210)]
    assert match(gt, pred) == (2, 1, 1)                        # 2 hits, 1 false box, 1 missed vehicle


def test_finetuned_decoder_maps_classes_back_to_coco():
    det = FinetunedDetector.__new__(FinetunedDetector)
    det.size, det.score_thr, det.nms_thr, det.input_name = 640, 0.3, 0.45, "images"
    out = np.zeros((1, 4 + 6, 3), np.float32)
    out[0, :4, 0] = (320, 320, 100, 80)                         # one box in the middle ...
    out[0, 4 + 1, 0] = 0.9                                      # ... class index 1 = motorcycle
    det.session = type("S", (), {"run": lambda self, *_: [out],
                                 "get_inputs": lambda self: [type("I", (), {"shape": [1, 3, "h", "w"]})()]})()
    res = det.detect(np.zeros((288, 352, 3), np.uint8))
    assert len(res) == 1 and res[0][0] == FT_TO_COCO[1] == 3
    assert dets_to_ft(res)[0][0] == 1


def test_ensemble_and_tiling():
    import numpy as np
    from cctv.ensemble import EnsembleDetector, TiledDetector, nms

    class Fake:
        def __init__(self, dets): self.dets, self.score_thr = dets, 0.3
        def detect(self, img): return [d for d in self.dets if d[1] >= self.score_thr]

    a = Fake([(2, 0.9, 10, 10, 50, 50), (7, 0.8, 100, 100, 200, 200)])
    b = Fake([(2, 0.7, 12, 12, 52, 52), (3, 0.9, 300, 200, 320, 240)])
    ens = EnsembleDetector([a, b])
    res = ens.detect(np.zeros((288, 352, 3), np.uint8))
    classes = sorted(r[0] for r in res)
    assert classes == [2, 3, 7]                      # the two cars fuse into one; single-model boxes survive
    car = [r for r in res if r[0] == 2][0]
    assert abs(car[1] - 0.8) < 1e-6                  # mean of both scores
    assert len(nms([(2, 0.9, 0, 0, 10, 10), (2, 0.5, 1, 1, 10, 10)])) == 1

    class Corner(Fake):                              # reports a car only in the crop that sees the bottom-right corner
        def detect(self, img):
            return [(2, 0.9, 5.0, 5.0, 25.0, 25.0)] if img.shape[0] < 288 else []
    tiled = TiledDetector(Corner([]), grid=2)
    out = tiled.detect(np.zeros((288, 352, 3), np.uint8))
    assert out and all(0 <= d[2] < 352 and 0 <= d[3] < 288 for d in out)
