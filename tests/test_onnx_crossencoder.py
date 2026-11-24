import math
import os

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from tokenizers.processors import TemplateProcessing  # noqa: E402

from decide.backends import load  # noqa: E402
from decide.backends.onnx_crossencoder import OnnxCrossEncoderBackend  # noqa: E402
from decide.errors import ConfigError  # noqa: E402
from decide.types import Noul, Request  # noqa: E402

WORDS = ["hello", "world", "yes", "no", "charged", "twice", "question", "refund"]
VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{w: i + 4 for i, w in enumerate(WORDS)}}
MAX_LENGTH = 8


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@pytest.fixture
def bundle(tmp_path):
    tok = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(f'{{"max_length": {MAX_LENGTH}}}')

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    nodes = [
        helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"]),
        helper.make_node("Cast", ["masked"], ["masked_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["masked_f", "axes"], ["logits"], keepdims=1),
    ]
    graph = helper.make_graph(nodes, "sum_ids", [ids, mask], [out], initializer=[axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(tmp_path / "model.onnx"))
    return tmp_path


def encoded_sum(bundle, query, doc, limit=MAX_LENGTH):
    tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    return sum(tok.encode(query, doc).ids[:limit])


def noul_request(state):
    return Request(state, {"n": Noul("question")})


def test_noul_is_sigmoid_of_the_summed_ids(bundle):
    backend = OnnxCrossEncoderBackend(bundle)
    r = backend.decide(noul_request("hello world"))
    query, doc = r.meta.raw["pairs"][0]
    expected = encoded_sum(bundle, query, doc)
    assert abs(r.nouls["n"].noul - sigmoid(expected)) < 1e-6


def test_long_pairs_are_truncated_to_the_bundle_max_length(bundle):
    state = " ".join(["world"] * 100)
    r = OnnxCrossEncoderBackend(bundle).decide(noul_request(state))
    query, doc = r.meta.raw["pairs"][0]
    tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    assert len(tok.encode(query, doc).ids) > MAX_LENGTH
    tok.enable_truncation(max_length=MAX_LENGTH)
    ids = tok.encode(query, doc).ids
    assert len(ids) == MAX_LENGTH
    assert r.meta.raw["logits"][0] == sum(ids)
    assert abs(r.nouls["n"].noul - sigmoid(sum(ids))) < 1e-6


def test_unicode_and_markup_score(bundle):
    r = OnnxCrossEncoderBackend(bundle).decide(
        noul_request("<@U123> :tada: 日本語 <https://x|link>")
    )
    assert 0 < r.nouls["n"].noul < 1


def test_batch_matches_single(bundle):
    backend = OnnxCrossEncoderBackend(bundle)
    requests = [noul_request(s) for s in ("hello", "hello world", "charged twice refund")]
    batch = backend.decide_batch(requests)
    singles = [backend.decide(req) for req in requests]
    assert [r.nouls["n"].noul for r in batch] == [r.nouls["n"].noul for r in singles]


def test_batch_size_chunks_do_not_change_scores(bundle):
    requests = [noul_request(s) for s in ("hello", "hello world", "charged twice refund")]
    small = OnnxCrossEncoderBackend(bundle, batch_size=1).decide_batch(requests)
    large = OnnxCrossEncoderBackend(bundle, batch_size=32).decide_batch(requests)
    assert [r.nouls["n"].noul for r in small] == [r.nouls["n"].noul for r in large]


def test_load_by_registry_name(bundle):
    backend = load("onnx_crossencoder", model_dir=str(bundle))
    assert isinstance(backend, OnnxCrossEncoderBackend)


def test_load_without_extra_names_the_extra(monkeypatch):
    import builtins

    real = builtins.__import__

    def deny(name, *a, **kw):
        if name.split(".")[0] in {"onnxruntime", "tokenizers"}:
            raise ImportError(name)
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", deny)
    with pytest.raises(ConfigError, match=r"pydecide\[onnx\]"):
        load("onnx_crossencoder", model_dir="unused")


def test_missing_bundle_file_is_config_error(bundle):
    os.remove(bundle / "config.json")
    with pytest.raises(ConfigError, match="config.json"):
        OnnxCrossEncoderBackend(bundle)


def test_session_uses_one_thread(bundle):
    options = OnnxCrossEncoderBackend(bundle)._session.get_session_options()
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1


@pytest.mark.live
def test_live_minilm_bundle_matches_sentence_transformers(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer

    from decide.backends.crossencoder import CrossEncoderBackend

    name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    st_model = CrossEncoder(name, device="cpu")
    hf = st_model.model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    enc = tokenizer("a", "b", return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(
        hf,
        tuple(enc[n] for n in names),
        str(tmp_path / "model.onnx"),
        input_names=names,
        output_names=["logits"],
        dynamo=False,
        dynamic_axes={**{n: {0: "batch", 1: "seq"} for n in names}, "logits": {0: "batch"}},
    )
    tokenizer.backend_tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text('{"max_length": 512}')

    req = Request(
        "I was charged twice for my subscription",
        {"n": Noul("Is this a billing problem?")},
    )
    expected = CrossEncoderBackend(st_model).decide(req).nouls["n"].noul
    actual = OnnxCrossEncoderBackend(tmp_path).decide(req).nouls["n"].noul
    assert abs(actual - expected) < 1e-4
