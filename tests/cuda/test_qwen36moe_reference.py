"""Qwen3.6-35B-A3B's fp32 reference forward (``qwen3_5_moe/cuda/reference.py``).

On tiny random weights (CPU, no checkpoint): cached serial decode equals the full-sequence forward, for the model and
the MTP head; the sliced head scores equal full logits; router ties go to the lower id.

On the real checkpoint (skipped when it is not in the Hugging Face cache; run in NVIDIA's container with
``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0``, else fp32 matmuls run in TF32): teacher-forced NLL and top-1 over public
text, greedy replies to three chat prompts (thinking off), and the MTP drafter's top-1 agreement with the model's
next token, which checks the order of the drafter's fc input and which hidden state it reads.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from tensorfold.families import qwen3_5_moe as family
from tensorfold.families.qwen3_5_moe.cuda import reference as R
from tensorfold.families.qwen3_5_moe.cuda import weights as W

BF, F32 = torch.bfloat16, torch.float32


# ---------------------------------------------------------------------------------------------------------------
# tiny random weights


def _config() -> W.Config:
    return W.Config(
        hidden=128, layers=4, layer_types=["linear", "attention", "linear", "attention"], vocab=320, eps=1e-6,
        heads=4, kv_heads=2, head_dim=64, attn_gate=True, rope_theta=1e7, partial_rotary=0.25, rotary_dim=16,
        mrope_section=(3, 3, 2), nk=2, nv=4, dk=32, dv=32, conv_kernel=4, experts=8, top_k=2, moe_width=64,
        shared_width=64, norm_topk=True, tie_embeddings=False, mtp_layers=1, max_position=4096, eos=(2,), bits=4,
        group_size=64)


def _qw(gen: torch.Generator, n: int, k: int, lead: tuple[int, ...] = (), gain: float = 1.0) -> W.QW:
    """Random 4-bit g64 weights whose values are near uniform with variance gain^2 / k."""

    words = torch.randint(-2 ** 31, 2 ** 31, (*lead, n, k // 8), generator=gen, dtype=torch.int64).to(torch.int32)
    step = gain * (12.0 / k) ** 0.5 / 15
    scales = (step * (0.5 + torch.rand((*lead, n, k // 64), generator=gen))).to(BF)
    biases = (-7.5 * scales.float() + 0.1 * step * torch.randn((*lead, n, k // 64), generator=gen)).to(BF)
    return W.QW(words, scales, biases, 4, 64)


def _scale(gen: torch.Generator, n: int) -> torch.Tensor:
    return 1.0 + 0.1 * torch.randn(n, generator=gen)


def _moe(gen: torch.Generator, c: W.Config) -> W.MoEW:
    e, d, m = c.experts + 1, c.hidden, c.moe_width
    router = 3.0 * torch.randn((e, d), generator=gen) / d ** 0.5
    return W.MoEW(router, _qw(gen, m, d, (e,)), _qw(gen, m, d, (e,)), _qw(gen, d, m, (e,)))


def _attn(gen: torch.Generator, c: W.Config) -> W.AttnW:
    return W.AttnW(_qw(gen, sum(c.attn_rows), c.hidden), _scale(gen, c.head_dim), _scale(gen, c.head_dim),
                   _qw(gen, c.hidden, c.heads * c.head_dim))


def _gdn(gen: torch.Generator, c: W.Config) -> W.GDNW:
    conv = (0.5 * torch.randn((c.conv_dim, c.conv_kernel), generator=gen)).to(BF)
    return W.GDNW(_qw(gen, sum(c.gdn_rows), c.hidden), conv, torch.log(1.0 + 15.0 * torch.rand(c.nv, generator=gen)),
                  torch.randn(c.nv, generator=gen), _scale(gen, c.dv).to(BF), _qw(gen, c.hidden, c.nv * c.dv))


def _tiny(seed: int = 0) -> tuple[W.Weights, W.MTPW]:
    gen = torch.Generator().manual_seed(seed)
    c = _config()
    layers = [W.LayerW(i, kind == "linear", _scale(gen, c.hidden), _scale(gen, c.hidden),
                       _gdn(gen, c) if kind == "linear" else None, None if kind == "linear" else _attn(gen, c),
                       _moe(gen, c)) for i, kind in enumerate(c.layer_types)]
    half = c.rotary_dim // 2
    inv = (torch.tensor(c.rope_theta, dtype=torch.float64) ** (-torch.arange(half, dtype=torch.float64) / half)).to(F32)
    w = W.Weights(c, _qw(gen, c.vocab, c.hidden, gain=8.0), layers, _scale(gen, c.hidden),
                  _qw(gen, c.vocab, c.hidden, gain=4.0), inv)
    mtp_layer = W.LayerW(0, False, _scale(gen, c.hidden), _scale(gen, c.hidden), None, _attn(gen, c), _moe(gen, c))
    mtp = W.MTPW(c, _scale(gen, c.hidden), _scale(gen, c.hidden), _qw(gen, c.hidden, 2 * c.hidden), mtp_layer,
                 _scale(gen, c.hidden))
    return w, mtp


@pytest.fixture
def small_blocks(monkeypatch):
    """Slice the head, the dequantization and the attention queries finely, so the tiny model crosses slices."""

    monkeypatch.setattr(R, "_HEAD_ROWS", 100)
    monkeypatch.setattr(R, "_DEQ_ROWS", 96)
    monkeypatch.setattr(R, "_QUERY_ROWS", 3)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("bf16, tol", [(False, 2e-5), (True, 5e-3)])
@pytest.mark.parametrize("schedule", [[3] + [1] * 10, [4, 1, 1, 5, 2]])
def test_cached_serial_decode_equals_the_full_forward(small_blocks, bf16, tol, schedule):
    w, _ = _tiny()
    tokens = torch.randint(0, w.cfg.vocab, (sum(schedule),), generator=torch.Generator().manual_seed(1))
    full_st = R.new_state(w, bf16=bf16)
    full = R.forward(w, tokens, full_st)
    st, parts, at = R.new_state(w, bf16=bf16), [], 0
    for n in schedule:
        parts.append(R.forward(w, tokens[at:at + n], st))
        at += n
    serial = torch.cat(parts)
    assert serial.shape == full.shape == (len(tokens), w.cfg.vocab)
    assert st.pos == full_st.pos == len(tokens)
    err = _rel(serial, full)
    print(f"\nbf16={bf16} schedule={schedule}: serial vs full logits, max rel err {err:.2e}")
    assert err < tol
    if not bf16:                                               # the caches end where one full pass leaves them
        for i, layer in enumerate(w.layers):
            if layer.linear:
                assert _rel(st.rec[i], full_st.rec[i]) < tol and _rel(st.conv[i], full_st.conv[i]) < tol
            else:
                assert _rel(st.kv[i][0], full_st.kv[i][0]) < tol and _rel(st.kv[i][1], full_st.kv[i][1]) < tol


def test_the_mtp_head_decodes_serially_as_it_runs_in_full(small_blocks):
    w, mtp = _tiny()
    tokens = torch.randint(0, w.cfg.vocab, (10,), generator=torch.Generator().manual_seed(2))
    h = R.hidden(w, tokens, R.new_state(w, bf16=False))
    full = R.mtp_hidden(mtp, w, tokens[1:], h[:-1], R.new_mtp_state(mtp, w, bf16=False))
    st = R.new_mtp_state(mtp, w, bf16=False)
    serial = torch.cat([R.mtp_hidden(mtp, w, tokens[1 + t:2 + t], h[t:t + 1], st) for t in range(len(tokens) - 1)])
    assert st.pos == len(tokens) - 1
    assert _rel(R.logits(w, serial), R.logits(w, full)) < 1e-5
    # the MTP head reads its inputs: another hidden state or another token changes its output
    assert _rel(R.mtp_hidden(mtp, w, tokens[1:], h[:-1].flip(0), R.new_mtp_state(mtp, w, bf16=False)), full) > 1e-2
    assert _rel(R.mtp_hidden(mtp, w, tokens[1:].flip(0), h[:-1], R.new_mtp_state(mtp, w, bf16=False)), full) > 1e-2


def test_sliced_head_scores_equal_full_logits(small_blocks):
    w, _ = _tiny()
    tokens = torch.randint(0, w.cfg.vocab, (9,), generator=torch.Generator().manual_seed(3))
    h = R.hidden(w, tokens, R.new_state(w))
    full = R.logits(w, h)
    targets = torch.randint(0, w.cfg.vocab, (9,), generator=torch.Generator().manual_seed(4))
    arg, nll = R.head_top1(w, h, targets)
    assert torch.equal(arg, full.argmax(1))
    want = -torch.log_softmax(full, 1).gather(1, targets[:, None])[:, 0]
    assert torch.allclose(nll, want, rtol=1e-5, atol=1e-5)
    assert R.head_top1(w, h)[1] is None


def test_router_ties_go_to_the_lower_id():
    w, _ = _tiny()
    c, m = w.cfg, w.layers[0].moe
    router = torch.zeros_like(m.router)
    router[6, 0] = router[1, 0] = 5.0                          # experts 1 and 6 tie for first
    router[[2, 5], 0] = 2.0                                    # 2 and 5 tie below them
    x = torch.zeros((3, c.hidden))
    x[:, 0] = 1.0
    ids, top = R.route(W.MoEW(router, m.gate, m.up, m.down), x, c)
    assert ids.tolist() == [[1, 6]] * 3
    assert torch.allclose(top, torch.full_like(top, 0.5))
    ids, _ = R.route(W.MoEW(router, m.gate, m.up, m.down), x, W.Config(**{**c.__dict__, "top_k": 4}))
    assert ids.tolist() == [[1, 6, 2, 5]] * 3


# ---------------------------------------------------------------------------------------------------------------
# the real checkpoint


def _cached(repo: str):
    from tensorfold import hub

    try:
        found = hub.cached(repo)
    except Exception:  # noqa: BLE001 - no huggingface_hub or no cache
        found = None
    if found is None or not (found / "model.safetensors").is_file() and \
            not (found / "model.safetensors.index.json").is_file():
        pytest.skip(f"{repo} is not in the Hugging Face cache")
    index = found / "model.safetensors.index.json"
    if index.is_file():
        import json

        if not all((found / s).is_file() for s in set(json.loads(index.read_text())["weight_map"].values())):
            pytest.skip(f"{repo}'s shards are not all in the Hugging Face cache")
    return found


# Public text: the first chapter of Jane Austen's Pride and Prejudice (1813) and the United States Declaration of
# Independence (1776), both in the public domain, and a passage written for this test.
AUSTEN = """\
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a
wife.

However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is
so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or
other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she told me all about it."

Mr. Bennet made no answer.

"Do you not want to know who has taken it?" cried his wife impatiently.

"You want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the
north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with
it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his
servants are to be in the house by the end of next week."

"What is his name?"

"Bingley."

"Is he married or single?"

"Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for
our girls!"

"How so? How can it affect them?"

"My dear Mr. Bennet," replied his wife, "how can you be so tiresome! You must know that I am thinking of his
marrying one of them."

"Is that his design in settling here?"

"Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and
therefore you must visit him as soon as he comes."

"I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be
still better, for as you are as handsome as any of them, Mr. Bingley may like you the best of the party."

"My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary
now. When a woman has five grown-up daughters, she ought to give over thinking of her own beauty."

"In such cases, a woman has not often much beauty to think of."

"But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood."

"It is more than I engage for, I assure you."

"But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady
Lucas are determined to go, merely on that account, for in general, you know, they visit no newcomers. Indeed you
must go, for it will be impossible for us to visit him if you do not."

"You are over-scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines
by you to assure him of my hearty consent to his marrying whichever he chooses of the girls; though I must throw in
a good word for my little Lizzy."

"I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so
handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference."

"They have none of them much to recommend them," replied he; "they are all silly and ignorant like other girls; but
Lizzy has something more of quickness than her sisters."

"Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no
compassion for my poor nerves."

"You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention
them with consideration these last twenty years at least."

"Ah, you do not know what I suffer."

"But I hope you will get over it, and live to see many young men of four thousand a year come into the
neighbourhood."

"It will be no use to us, if twenty such should come, since you will not visit them."

"Depend upon it, my dear, that when there are twenty, I will visit them all."

Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of
three-and-twenty years had been insufficient to make his wife understand his character. Her mind was less difficult
to develop. She was a woman of mean understanding, little information, and uncertain temper. When she was
discontented, she fancied herself nervous. The business of her life was to get her daughters married; its solace was
visiting and news."""

DECLARATION = """\
When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have
connected them with another, and to assume among the powers of the earth, the separate and equal station to which
the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they
should declare the causes which impel them to the separation.

We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with
certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. That to secure these
rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, That
whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to
abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in
such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate
that Governments long established should not be changed for light and transient causes; and accordingly all
experience hath shewn, that mankind are more disposed to suffer, while evils are sufferable, than to right
themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations,
pursuing invariably the same Object evinces a design to reduce them under absolute Despotism, it is their right, it
is their duty, to throw off such Government, and to provide new Guards for their future security. Such has been the
patient sufferance of these Colonies; and such is now the necessity which constrains them to alter their former
Systems of Government. The history of the present King of Great Britain is a history of repeated injuries and
usurpations, all having in direct object the establishment of an absolute Tyranny over these States. To prove this,
let Facts be submitted to a candid world.

He has refused his Assent to Laws, the most wholesome and necessary for the public good.

He has forbidden his Governors to pass Laws of immediate and pressing importance, unless suspended in their
operation till his Assent should be obtained; and when so suspended, he has utterly neglected to attend to them.

He has refused to pass other Laws for the accommodation of large districts of people, unless those people would
relinquish the right of Representation in the Legislature, a right inestimable to them and formidable to tyrants
only.

He has called together legislative bodies at places unusual, uncomfortable, and distant from the depository of their
public Records, for the sole purpose of fatiguing them into compliance with his measures.

He has dissolved Representative Houses repeatedly, for opposing with manly firmness his invasions on the rights of
the people.

He has refused for a long time, after such dissolutions, to cause others to be elected; whereby the Legislative
powers, incapable of Annihilation, have returned to the People at large for their exercise; the State remaining in
the mean time exposed to all the dangers of invasion from without, and convulsions within.

He has endeavoured to prevent the population of these States; for that purpose obstructing the Laws for
Naturalization of Foreigners; refusing to pass others to encourage their migrations hither, and raising the
conditions of new Appropriations of Lands.

He has obstructed the Administration of Justice, by refusing his Assent to Laws for establishing Judiciary powers.

He has made Judges dependent on his Will alone, for the tenure of their offices, and the amount and payment of their
salaries.

He has erected a multitude of New Offices, and sent hither swarms of Officers to harrass our people, and eat out
their substance.

He has kept among us, in times of peace, Standing Armies without the Consent of our legislatures.

He has affected to render the Military independent of and superior to the Civil power.

He has combined with others to subject us to a jurisdiction foreign to our constitution, and unacknowledged by our
laws; giving his Assent to their Acts of pretended Legislation:

For Quartering large bodies of armed troops among us:

For protecting them, by a mock Trial, from punishment for any Murders which they should commit on the Inhabitants of
these States:

For cutting off our Trade with all parts of the world:

For imposing Taxes on us without our Consent:

For depriving us in many cases, of the benefits of Trial by Jury:

For transporting us beyond Seas to be tried for pretended offences:

For abolishing the free System of English Laws in a neighbouring Province, establishing therein an Arbitrary
government, and enlarging its Boundaries so as to render it at once an example and fit instrument for introducing
the same absolute rule into these Colonies:

For taking away our Charters, abolishing our most valuable Laws, and altering fundamentally the Forms of our
Governments:

For suspending our own Legislatures, and declaring themselves invested with power to legislate for us in all cases
whatsoever.

He has abdicated Government here, by declaring us out of his Protection and waging War against us.

He has plundered our seas, ravaged our Coasts, burnt our towns, and destroyed the lives of our people.

He is at this time transporting large Armies of foreign Mercenaries to compleat the works of death, desolation and
tyranny, already begun with circumstances of Cruelty and perfidy scarcely paralleled in the most barbarous ages, and
totally unworthy the Head of a civilized nation.

He has constrained our fellow Citizens taken Captive on the high Seas to bear Arms against their Country, to become
the executioners of their friends and Brethren, or to fall themselves by their Hands.

In every stage of these Oppressions We have Petitioned for Redress in the most humble terms: Our repeated Petitions
have been answered only by repeated injury. A Prince whose character is thus marked by every act which may define a
Tyrant, is unfit to be the ruler of a free people.

Nor have We been wanting in attentions to our British brethren. We have warned them from time to time of attempts by
their legislature to extend an unwarrantable jurisdiction over us. We have reminded them of the circumstances of our
emigration and settlement here. We have appealed to their native justice and magnanimity, and we have conjured them
by the ties of our common kindred to disavow these usurpations, which would inevitably interrupt our connections and
correspondence. They too have been deaf to the voice of justice and of consanguinity. We must, therefore, acquiesce
in the necessity, which denounces our Separation, and hold them, as we hold the rest of mankind, Enemies in War, in
Peace Friends.

We, therefore, the Representatives of the united States of America, in General Congress, Assembled, appealing to the
Supreme Judge of the world for the rectitude of our intentions, do, in the Name, and by Authority of the good People
of these Colonies, solemnly publish and declare, That these United Colonies are, and of Right ought to be Free and
Independent States; that they are Absolved from all Allegiance to the British Crown, and that all political
connection between them and the State of Great Britain, is and ought to be totally dissolved; and that as Free and
Independent States, they have full Power to levy War, conclude Peace, contract Alliances, establish Commerce, and to
do all other Acts and Things which Independent States may of right do. And for the support of this Declaration, with
a firm reliance on the protection of divine Providence, we mutually pledge to each other our Lives, our Fortunes and
our sacred Honor."""

LIGHTHOUSE = """\
A lighthouse has one job, and it has done that job in much the same way for more than two thousand years: it tells a
sailor where the land is before the land can do any harm. The earliest lights were open fires kept burning on
hilltops or on the roofs of towers, and the keeper's work was mostly the hauling of wood or coal up a long flight of
stairs. A fire gives a poor light, though. Most of it goes up into the sky or back towards the land, where no ship
will ever see it, and in rain or fog it can vanish from a mile away.

The great change came in the early nineteenth century, when the French engineer Augustin Fresnel worked out how to
bend light with a ring of glass prisms. Instead of one enormous and impossibly heavy lens, he built a lens from many
thin concentric pieces, each shaped to catch light from a single lamp and send it out in a narrow, nearly level
beam. A lamp that had once been visible for a few miles could now be seen from twenty or more. The same idea is
still used in car headlights, overhead projectors and the flat plastic magnifiers sold in bookshops.

A beam alone is not enough, because a sailor who sees a light at night also needs to know which light it is. For
that reason every lighthouse on a coast is given its own character: a pattern of flashes, eclipses and colours that
repeats at a fixed interval. One light might show a single white flash every five seconds, its neighbour two red
flashes every ten. The patterns are printed on charts and in published lists, so a navigator who counts the seconds
between flashes can find the light's name, and from it the ship's position.

Today most lighthouses run without keepers. Electric lamps, solar panels and batteries have replaced the oil and the
clockwork that once turned the lens, and satellite navigation tells a ship's crew where they are to within a few
metres. Yet the lights are still maintained, and many sailors are glad of them. Electronics fail, and on a dark
night close to a rocky shore, a flash of light in the right place at the right time is a very reassuring thing to
see."""



def _unwrap(text: str) -> str:
    """Paragraphs as single lines (the passages are wrapped here to fit the line length)."""

    return "\n\n".join(" ".join(p.split("\n")) for p in text.split("\n\n"))


PASSAGES = {"austen": _unwrap(AUSTEN), "declaration": _unwrap(DECLARATION), "lighthouse": _unwrap(LIGHTHOUSE)}

PROMPTS = [
    ("What is the capital of France? Answer in one sentence.", ["paris"]),
    ("Name the three primary colours of light, separated by commas.", ["red", "green", "blue"]),
    ("In one sentence, explain why the sky looks blue during the day.", ["scatter"]),
]


def _tf32_off() -> None:
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        pytest.skip("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 runs fp32 matmuls in TF32: set it to 0 for the reference")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@pytest.fixture(scope="module")
def real():
    snap = _cached(family.MODELS[0])
    _tf32_off()
    from tokenizers import Tokenizer

    t0 = time.time()
    w = W.load(snap, "cuda")
    print(f"\nloaded {snap.name[:12]} in {time.time() - t0:.0f} s: {w.nbytes() / 2 ** 30:.2f} GiB on the GPU")
    yield snap, w, Tokenizer.from_file(str(snap / "tokenizer.json"))
    del w
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def scored(real):
    """Per passage: the token ids, the hidden states after the final norm, the model's argmax at each position and
    the NLL of each next token."""

    _, w, tok = real
    out = {}
    for name, text in PASSAGES.items():
        ids = tok.encode(text, add_special_tokens=False).ids
        t0 = time.time()
        st = R.new_state(w)
        pre = R.hidden(w, torch.tensor(ids), st, normed=False)
        h = R.final_norm(w, pre, st)
        arg, nll = R.head_top1(w, h, torch.tensor(ids[1:] + [0]))
        out[name] = (ids, h, arg, nll[:-1], time.time() - t0, pre)
    return out


def test_teacher_forced_nll_and_top1_over_public_text(real, scored):
    """NLL and top-1 with activations rounded to bf16 where the kernels store them (the default), and with every
    activation fp32: the rounding should cost next to nothing."""

    _, w, _ = real
    total = {"bf16": [0.0, 0], "fp32": [0.0, 0]}
    same, n_all = 0, 0
    print()
    for name, (ids, _, arg, nll, seconds, _) in scored.items():
        target = torch.tensor(ids[1:])
        h32 = R.hidden(w, torch.tensor(ids), R.new_state(w, bf16=False))
        arg32, nll32 = R.head_top1(w, h32[:-1], target)
        n = len(ids) - 1
        for mode, a, l in (("bf16", arg[:-1], nll), ("fp32", arg32, nll32)):
            hit = int((a.cpu() == target).sum())
            total[mode][0] += float(l.sum())
            total[mode][1] += hit
            print(f"{name:12s} {mode} {n:5d} tokens  NLL {float(l.mean()):.4f}  ppl {float(l.mean().exp()):7.3f}  "
                  f"top-1 {hit / n:.4f}" + (f"  ({seconds:.0f} s)" if mode == "bf16" else ""))
        same += int((arg[:-1] == arg32).sum())
        n_all += n
    for mode, (nll_sum, hits) in total.items():
        print(f"{'all':12s} {mode} {n_all:5d} tokens  NLL {nll_sum / n_all:.4f}  "
              f"ppl {torch.tensor(nll_sum / n_all).exp():7.3f}  top-1 {hits / n_all:.4f}")
    print(f"bf16 and fp32 activations pick the same top-1 at {same / n_all:.4f} of positions")
    mean = total["bf16"][0] / n_all
    assert n_all > 2000
    assert mean < 1.5 and total["bf16"][1] / n_all > 0.65
    assert abs(mean - total["fp32"][0] / n_all) < 0.02 and same / n_all > 0.97


def test_greedy_replies_to_chat_prompts_are_coherent(real):
    from tensorfold.cuda.server import ChatTemplate

    snap, w, tok = real
    template = ChatTemplate(snap)
    print()
    for prompt, must in PROMPTS:
        ids = tok.encode(template.render([{"role": "user", "content": prompt}], tools=None, enable_thinking=False),
                         add_special_tokens=False).ids
        t0 = time.time()
        out = R.greedy(w, ids, 64, w.cfg.eos)
        reply = tok.decode(out, skip_special_tokens=True).strip()
        print(f"Q: {prompt}\nA: {reply!r}  ({len(out)} tokens, {time.time() - t0:.0f} s)")
        assert out[-1] in w.cfg.eos, "the reply should end within 64 tokens"
        assert all(word in reply.lower() for word in must), reply
        # the cached decode picked what one full forward over prompt + reply picks, near-ties aside: the bf16
        # roundings land differently in a 1-row and a many-row pass (fp32 activations agree to 1e-5, tiny tests)
        full = R.forward(w, torch.tensor(ids + out[:-1]), R.new_state(w))[len(ids) - 1:]
        picked = full.gather(1, torch.tensor(out, device=full.device)[:, None])[:, 0]
        gap = full.max(1).values - picked
        print(f"   full forward vs cached decode: {int((full.argmax(1).cpu() != torch.tensor(out)).sum())} of "
              f"{len(out)} differ, largest gap {float(gap.max()):.4f}")
        assert float(gap.max()) < 0.25


def _swap_fc_halves(fc: W.QW, hidden: int) -> W.QW:
    """fc with its two input halves exchanged: what reading [hidden | embedding] would compute."""

    def swap(t: torch.Tensor, per: int) -> torch.Tensor:
        cut = hidden // per
        return torch.cat([t[:, cut:], t[:, :cut]], dim=1).contiguous()

    return W.QW(swap(fc.words, 32 // fc.bits), swap(fc.scales, fc.group), swap(fc.biases, fc.group), fc.bits,
                fc.group)


def test_the_mtp_head_agrees_with_the_models_next_token(real, scored):
    import dataclasses

    _, w, _ = real
    mtp = W.load_mtp(_cached(family.DRAFTER), w.cfg, "cuda")
    swapped = dataclasses.replace(mtp, fc=_swap_fc_halves(mtp.fc, w.cfg.hidden))
    rates: dict[str, list[tuple[int, int, int]]] = {"fc [embedding | hidden]": [], "fc [hidden | embedding]": [],
                                                   "hidden before the final norm": []}
    print()
    for name, (ids, h, arg, _, _, pre) in scored.items():
        # row t reads token t+1 and the hidden at t, and predicts token t+2; the model's own guess for token t+2 is
        # its argmax at position t+1
        for label, head, hs in (("fc [embedding | hidden]", mtp, h), ("fc [hidden | embedding]", swapped, h),
                                ("hidden before the final norm", mtp, pre)):
            hm = R.mtp_hidden(head, w, torch.tensor(ids[1:]), hs[:-1], R.new_mtp_state(head, w))
            guess, _ = R.head_top1(w, hm[:-1])
            agree = int((guess == arg[1:-1]).sum())
            right = int((guess.cpu() == torch.tensor(ids[2:])).sum())
            rates[label].append((agree, right, len(ids) - 2))
            print(f"{name:12s} {label:30s} agrees with the model {agree / (len(ids) - 2):.4f}, "
                  f"top-1 on the text {right / (len(ids) - 2):.4f}")
    summary = {}
    for label, r in rates.items():
        n = sum(k for _, _, k in r)
        summary[label] = (sum(a for a, _, _ in r) / n, sum(b for _, b, _ in r) / n)
    for label, (agree, right) in summary.items():
        print(f"{'all':12s} {label:30s} agrees with the model {agree:.4f}, top-1 on the text {right:.4f}")
    main = summary["fc [embedding | hidden]"][0]
    assert main > 0.5
    assert main > summary["fc [hidden | embedding]"][0] + 0.2
    assert main > summary["hidden before the final norm"][0]

